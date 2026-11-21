# YOLO26s small-deer refinement: prepared, not trained

- Status: prepared and launched on greglinux; final metrics pending.
- Question: Can training crops and a short low-LR fine-tune improve small-deer recall at a practical human-review workload?
- Evidence motivating it: [Modal comparison and crop pilot](2026-11-10-deer-v3-modal-comparison.md). The pilot improved one evaluation slice with extra inference passes; this does not establish that training crops will help.
- Entry points: [notebook and CLI](../../../prototyping/yolo26s-refinement/README.md), [recipe](../../../prototyping/yolo26s-refinement/recipe.json).

## Prepared recipe

Same original Deer v3 3,193/358/238 split, single deer class, 4x training-decoy weighting. No LVIS additions or evaluation-image mining. Initialize from the saved Modal YOLO26s deer checkpoint, SHA256 `a6374c47b5d1b97c8e4f8f16f586463a626de7a8a5eb1dbcdfd51bb0b3d1f672`. This is a new fine-tune, not an optimizer resume of Modal training.

Add 60%-width/height crops around small training deer, maximum 100 per source. Keep all intersecting deer labels, clipped to the crop; discard an entire crop if it creates an ambiguous sliver. Local preparation produced **467 crops / 1,777 boxes**: 100 each from Sherbrooke, Young Deer, BAMBI and Rucervus, 33 AWIR and 34 decoys. These are derived views, not independent new observations. The effective training list grows from 3,610 to 4,077 entries. Original images, labels and split assignments are unchanged; generated runtime path files were refreshed for this computer.

Use 1280, AdamW LR 0.0001, up to 25 epochs, patience 7 and mild augmentation. Initial laptop recipe uses batch 1/workers 0/FP32. These are hypotheses and conservative resource choices, not proven improvements or a solution to driver/hardware faults. Native checkpoint selection uses validation fitness. Test inference remains a separate explicit command.

In addition to existing F1 calibration, select a validation-only human-review threshold: target mean source recall 95% under two false boxes/image. Report target infeasibility explicitly and preserve source slices. This is a proposed workload assumption; tune it with reviewers before deployment. Historical test has already informed development and is a regression set, not a new unbiased flight test.

## Verification and artifacts

Local actual-data check confirmed all 467 crops use training images only, include every intersecting ground-truth box and round-trip their normalized YOLO coordinates within 0.0001 pixel. Two CPU tests passed for clipped-label completeness and infeasible-recall/locked-test-threshold behavior. Both notebooks validate and their code cells compile; frozen-v3 metadata checks passed. A contact sheet was visually inspected. This is annotation-transformation QA, not a complete semantic re-review of the source labels.

Local ignored preparation: `artifacts/yolo26s_refinement_preflight_v2/`. It contains the tile preview/index, generated training list, metadata and code snapshot. No weights or dataset files added to Git. A server preparation should be regenerated there because paths are absolute. The runner saves code revision/dirty state, recipe, prepared-file hashes and environment when used. Working-tree changes are uncommitted at preparation time.

## Flight implications and remaining work

The reported VTOL/gimbal/zoom plan increases the importance of oblique and distant views, frame timing, calibration and realistic optical/video conditions. Synthetic crops cannot supply that missing domain coverage. Collect a held-out camera session with dry/green ground, multiple ranges and zoom settings, blur and real negative objects. Do not relabel coyotes as deer or assume they caused most prior errors.

Map interfaces remain a cross-team discussion; no aircraft integration is implemented. The server subsequently became reachable. The GPU had no training processes; the only listed compute client was GNOME remote desktop (~140 MiB), which is explicitly allowed below 256 MiB by the idle check. All other compute processes still block launch.

## Server launch

- Code at launch: `1687177`, branch `experiment/yolo26s-refinement`; separate checkout `/home/mrd/yolo26s_refinement/code`.
- Data: `/home/mrd/deer-data/Deer Training Dataset v3`; canonical split, COCO and YOLO label hashes verified by preparation. No re-download.
- Original weights copied to `/home/mrd/yolo26s-deer-v3-best.pt` and checksum verified.
- Runtime: existing Python environment `/home/mrd/code/computer-vision-2026-v4/.venv/bin/python`, Ultralytics 8.4.155, PyTorch 2.8.0+cu128. Dependencies were not modified.
- Outputs: `/home/mrd/yolo26s_refinement/run1`; console log `/home/mrd/yolo26s_refinement/pipeline.log`.
- Detached user service: `deer-yolo26s-refinement-v1.service`. Sequence: smoke → fine-tune → validation and threshold calibration → historical-test regression. Stops on failure; no automatic GPU reset or repeated retries. No other models launched.
- Launch evidence and final status should be distinguished: service activation alone does not establish a successful run. Final completion requires weights, validation/test reports and `pipeline_complete.json`.

GPU smoke completed successfully and saved best/last checkpoints. Smoke best hash: `418be71ac590627748170b5767e4b8a97600fb1ec76ff8e57edc5f9d4e8bedb0`. The full run was then observed advancing through epoch 1/25 (4,077 steps), with trainer-reported GPU memory about 2.6 GB and no CUDA errors at the handoff check. This is launch validation, not a completed accuracy result. The notebook's defaults point at this deployment and include a read-only console-progress cell; leave its training toggle off while the service is running.
