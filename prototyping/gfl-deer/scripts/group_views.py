"""Conservative SIFT geometric matching for near-overlapping decoy crops."""
import json
from pathlib import Path
import cv2
import numpy as np
from deer_pipeline.common import ROOT,read_json,write_json
rs=[r for r in read_json(ROOT/'data/decoys/manifest.json') if r['status']=='reviewed']
parent=list(range(len(rs)))
def find(x):
    while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
    return x
def union(i,j):parent[find(j)]=find(i)
sift=cv2.SIFT_create(nfeatures=220);features=[]
for r in rs:
    im=cv2.imread(r['file_name'],0);im=cv2.resize(im,(max(1,round(im.shape[1]*min(1,600/max(im.shape)))),max(1,round(im.shape[0]*min(1,600/max(im.shape))))))
    kp,desc=sift.detectAndCompute(im,None);features.append((np.array([p.pt for p in kp],dtype=np.float32),desc))
matcher=cv2.BFMatcher();links=[]
for i,(points,desc) in enumerate(features):
    if desc is None:continue
    for j in range(i):
        if find(i)==find(j):continue
        other,desc2=features[j]
        if desc2 is None or len(desc2)<2:continue
        good=[a for a,b in matcher.knnMatch(desc,desc2,k=2) if a.distance<.65*b.distance]
        if len(good)<12:continue
        a=np.float32([points[m.queryIdx] for m in good]);b=np.float32([other[m.trainIdx] for m in good])
        H,mask=cv2.findHomography(a,b,cv2.RANSAC,3.)
        if H is not None and int(mask.sum())>=12 and mask.mean()>=.85:
            union(i,j);links.append(dict(a=rs[i]['source_id'],b=rs[j]['source_id'],inliers=int(mask.sum())))
    if i%40==0:print(i,'/',len(rs),'links',len(links),flush=True)
mapping={r['source_id']:'decoy_sift_'+str(find(i)) for i,r in enumerate(rs)}
write_json(ROOT/'reports/decoy_view_groups.json',dict(groups=mapping,links=links,method='SIFT220/600px ratio0.65 homography3px >=12 inliers >=85%'))
print('groups',len(set(mapping.values())))
