# YOLO26s small-deer refinement: prepared, not trained

- Status: CPU preparation checked; GPU smoke/training/evaluation not run.
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

Map interfaces remain a cross-team discussion; no aircraft integration is implemented. Michael's server was not reached because local Tailscale was stopped. No GPU work was launched. Next: review crop preview, restore connectivity, inspect server/GPU availability and run smoke before authorizing the full fine-tune.
