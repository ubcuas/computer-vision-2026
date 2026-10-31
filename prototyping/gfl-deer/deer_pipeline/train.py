"""MMDetection 2.28.2 training and inference; imports are lazy for local data QA."""
import copy
import os
import subprocess
import time
from pathlib import Path
from .common import ROOT,read_json,write_json
from .metrics import detection_metrics

MODELS={
'x101':('gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py','https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'),
'r50':('gfl_r50_fpn_1x_coco.py','https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth')}

def config(dataset,workdir,model='x101',lr=.00025,epochs=10,long_edge=2650,checkpoint=None,smoke=False,patience=0):
    import mmdet
    from mmcv import Config
    source=ROOT/'vendor'/'mmdetection'/'configs'/'gfl'/MODELS[model][0]
    if not source.exists():raise FileNotFoundError('Run scripts/setup_colab.sh to install the pinned source checkout')
    cfg=Config.fromfile(str(source));cfg.model.bbox_head.num_classes=1
    # Avoid an unnecessary secondary ImageNet download; load the complete COCO detector below.
    cfg.model.backbone.init_cfg=None
    cfg.load_from=checkpoint or MODELS[model][1]
    cfg.work_dir=str(Path(workdir).resolve());cfg.seed=42;cfg.gpu_ids=[0];cfg.device='cuda'
    scale=(long_edge,round(long_edge*1768/2650))
    trainpipe=[dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True),
      dict(type='Resize',img_scale=scale,keep_ratio=True),
      dict(type='RandomFlip',flip_ratio=[.25,.25],direction=['horizontal','vertical']),
      dict(type='Normalize',mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True),
      dict(type='MinimumSizePad',minimum=96),dict(type='Pad',size_divisor=32),dict(type='DefaultFormatBundle'),dict(type='Collect',keys=['img','gt_bboxes','gt_labels'])]
    testpipe=[dict(type='LoadImageFromFile'),dict(type='MultiScaleFlipAug',img_scale=scale,flip=False,
       transforms=[dict(type='Resize',keep_ratio=True),dict(type='RandomFlip'),
       dict(type='Normalize',mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True),
       dict(type='MinimumSizePad',minimum=96),dict(type='Pad',size_divisor=32),dict(type='ImageToTensor',keys=['img']),dict(type='Collect',keys=['img'])])]
    dataset=Path(dataset).resolve()
    for split in ['train','val','test']:
        cfg.data[split]=dict(type='CocoDataset',ann_file=str(dataset/'annotations'/f'{split}.json'),img_prefix='',
          classes=('deer',),pipeline=trainpipe if split=='train' else testpipe,filter_empty_gt=False)
    cfg.data.samples_per_gpu=1;cfg.data.workers_per_gpu=2
    cfg.optimizer=dict(type='SGD',lr=lr,momentum=.9,weight_decay=.0001)
    cfg.optimizer_config=dict(grad_clip=dict(max_norm=35,norm_type=2))
    cfg.lr_config=dict(policy='step',warmup='linear',warmup_iters=500,warmup_ratio=.001,step=[max(1,round(epochs*.6)),max(2,round(epochs*.8))])
    cfg.runner=dict(type='EpochBasedRunner',max_epochs=epochs)
    cfg.checkpoint_config=dict(interval=1,max_keep_ckpts=2)
    cfg.evaluation=dict(interval=1,metric='bbox',save_best='bbox_mAP')
    cfg.log_config=dict(interval=10,hooks=[dict(type='TextLoggerHook')])
    cfg.fp16=dict(loss_scale='dynamic');cfg.workflow=[('train',1)]
    cfg.model.test_cfg.score_thr=.001;cfg.model.test_cfg.max_per_img=300
    cfg.custom_hooks=[dict(type='ValidationF1Hook')]
    if patience:cfg.custom_hooks.append(dict(type='ValidationAPEarlyStoppingHook',patience=patience,priority=80))
    cfg.dist_params=dict(backend='nccl');cfg.log_level='INFO'
    if smoke:
        cfg.runner=dict(type='IterBasedRunner',max_iters=2)
        cfg.workflow=[('train',1)]
        cfg.evaluation=dict(interval=100,by_epoch=False,metric='bbox')
        cfg.checkpoint_config=dict(interval=1,by_epoch=False,max_keep_ckpts=2)
        cfg.lr_config=None;cfg.custom_hooks=[]
    return cfg

def early_stopping_state(previous,best_score,epoch,patience):
    """Track the EvalHook's cumulative best AP; ties do not reset patience."""
    previous=previous or {}
    best_epoch=epoch if best_score>previous.get('best_score',float('-inf')) else previous.get('best_epoch',epoch)
    return dict(best_score=best_score,best_epoch=best_epoch,epoch=epoch,
                patience=patience,stopped=epoch-best_epoch>=patience)

def register_padding():
    from mmdet.datasets.builder import PIPELINES
    import mmcv
    @PIPELINES.register_module(force=True)
    class MinimumSizePad:
        def __init__(self,minimum=96):self.minimum=minimum
        def __call__(self,results):
            for key in results.get('img_fields',['img']):
                im=results[key]
                results[key]=mmcv.impad(im,shape=(max(im.shape[0],self.minimum),max(im.shape[1],self.minimum)),pad_val=0)
            results['pad_shape']=results['img'].shape
            return results

def register_hooks():
    register_padding()
    import torch
    from mmcv.runner import HOOKS,Hook
    from mmdet.datasets import build_dataset,build_dataloader
    from mmdet.apis import single_gpu_test
    @HOOKS.register_module(force=True)
    class ValidationAPEarlyStoppingHook(Hook):
        def __init__(self,patience):self.patience=patience
        def after_train_epoch(self,runner):
            # EvalHook (priority 70) has updated its cumulative best AP before this hook.
            path=Path(runner.work_dir)/'early_stopping.json'
            previous=read_json(path) if path.exists() else None
            score=runner.meta.get('hook_msgs',{}).get('best_score',-1.)
            state=early_stopping_state(previous,float(score),runner.epoch+1,self.patience)
            write_json(path,state)
            if state['stopped']:
                runner.logger.info('Early stopping: validation AP has not improved for %s epochs',self.patience)
                runner._max_epochs=runner.epoch+1
    @HOOKS.register_module(force=True)
    class ValidationF1Hook(Hook):
        """Select an adaptation checkpoint with validation F1, never test predictions."""
        def after_train_epoch(self,runner):
            cfg=runner.meta['config_dict'];ds=build_dataset(cfg['data']['val'])
            loader=build_dataloader(ds,samples_per_gpu=1,workers_per_gpu=cfg['data']['workers_per_gpu'],dist=False,shuffle=False)
            result=single_gpu_test(runner.model,loader)
            gt=read_json(cfg['data']['val']['ann_file']);pred=results_to_coco(result,gt)
            from .metrics import calibrate
            cal=calibrate(gt,pred);metric=detection_metrics(gt,pred,cal['threshold'])
            runner.log_buffer.output['validation_f1']=metric['f1']
            folder=Path(runner.work_dir);bestpath=folder/'best_f1.json'
            best=read_json(bestpath)['f1'] if bestpath.exists() else -1
            if metric['f1']>best:
                runner.save_checkpoint(str(folder),filename_tmpl='best_f1.pth',create_symlink=False)
                write_json(bestpath,dict(f1=metric['f1'],epoch=runner.epoch+1,threshold=cal['threshold']))
                write_json(folder/'calibration.json',cal)
            runner.model.train()

def results_to_coco(results,gt):
    out=[]
    for im,classes in zip(gt['images'],results):
        for x1,y1,x2,y2,score in classes[0]:
            x1=max(0,min(float(x1),im['width']));y1=max(0,min(float(y1),im['height']))
            x2=max(0,min(float(x2),im['width']));y2=max(0,min(float(y2),im['height']))
            if x2>x1 and y2>y1:out.append(dict(image_id=im['id'],category_id=1,bbox=[x1,y1,x2-x1,y2-y1],score=float(score)))
    return out

def run_train(dataset,workdir,model='x101',lr=.00025,epochs=10,long_edge=2650,checkpoint=None,resume=None,smoke=False,patience=0):
    import torch,mmcv,mmdet
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.apis import train_detector
    if not torch.cuda.is_available():raise RuntimeError('Training requires the planned NVIDIA Colab runtime; no CPU training was started')
    register_hooks();cfg=config(dataset,workdir,model,lr,epochs,long_edge,checkpoint,smoke,patience)
    Path(workdir).mkdir(parents=True,exist_ok=True);cfg.dump(str(Path(workdir)/'config.py'))
    if resume:cfg.resume_from=str(resume)
    write_json(Path(workdir)/'environment.json',dict(torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,
       gpu=torch.cuda.get_device_name(),seed=42,model=model,resolution=long_edge,
       source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'vendor'/'mmdetection',text=True).strip()))
    (Path(workdir)/'pip-freeze.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True))
    set_random_seed(42,deterministic=True)
    modelobj=build_detector(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'));modelobj.init_weights()
    datasetobj=build_dataset(cfg.data.train);modelobj.CLASSES=('deer',)
    meta=dict(config=cfg.pretty_text,config_dict=cfg.to_dict(),CLASSES=('deer',),seed=42)
    train_detector(modelobj,[datasetobj],cfg,distributed=False,validate=not smoke,meta=meta)
    if smoke:
        path=Path(workdir)/'iter_2.pth'
        assert path.exists(),'Smoke test did not write a checkpoint'
        loaded=torch.load(path,map_location='cpu');assert 'optimizer' in loaded and 'state_dict'in loaded
        assert all(torch.isfinite(t).all() for t in loaded['state_dict'].values() if t.is_floating_point()), 'Non-finite checkpoint weights'
        write_json(Path(workdir)/'smoke_result.json',dict(backward_pass=True,checkpoint_roundtrip=True,gpu=torch.cuda.get_device_name()))

def predict(config_path,checkpoint,annotations,output,device='cuda:0'):
    import torch
    from mmdet.apis import init_detector,inference_detector
    from PIL import Image,ImageDraw
    register_padding()
    gt=read_json(annotations);model=init_detector(str(config_path),str(checkpoint),device=device)
    rows=[];times=[];preview=Path(output).parent/'examples';preview.mkdir(parents=True,exist_ok=True)
    for n,im in enumerate(gt['images']):
        if device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter();res=inference_detector(model,im['file_name'])
        if device.startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        if n:times.append(elapsed)
        boxes=results_to_coco([res],dict(images=[im]));rows.extend(boxes)
        if n<12:
            image=Image.open(im['file_name']).convert('RGB');d=ImageDraw.Draw(image)
            for b in boxes:
                if b['score']<.5:continue
                x,y,w,h=b['bbox'];d.rectangle((x,y,x+w,y+h),outline='red',width=3);d.text((x,y),f'{b["score"]:.2f}',fill='red')
            image.save(preview/(Path(im['file_name']).stem+'.jpg'))
    write_json(output,rows)
    write_json(str(output)+'.latency.json',dict(device=device,images=len(gt['images']),warmup_images=1,
        mean_seconds=sum(times)/len(times) if times else None,includes='single-image loading + inference + postprocessing; excludes output rendering'))
    return rows
