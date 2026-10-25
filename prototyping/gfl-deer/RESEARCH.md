# Research decision and replication ledger

Selected: GFL following Moreni, Theau & Foucher (2023), *Do you get what you see?*, [PLOS ONE](https://doi.org/10.1371/journal.pone.0284449).

| Candidate | Relevant result and limitation | Decision |
|---|---|---|
| Moreni GFL | F1 0.95-0.97 across summer/winter variants; validation-based best-of-five, no independent test | Reconstruct core method and add held-out decoy evaluation |
| Roca et al. aerial marsh deer | YOLO11n-seg improved recall/F1; high-resolution small-target evidence | Team already investigates YOLO; retain as reference |
| Lyu et al. thermal deer | AP50 91.6%; modality mismatch, original training data/code not located | Not today's primary |
| Goto et al. LT-DETR | Abstract reports better cross-survey performance than YOLO11; full methods not accessible in review | Promising but cannot faithfully reconstruct from abstract |
| Hjort-YOLO | Camera-trap focus; original deer data author-requested | Exclude from today's critical path |

Sources: [Roca](https://arxiv.org/abs/2506.00154), [Lyu](https://doi.org/10.1016/j.ecoinf.2023.102383), [Goto](https://www.sciencedirect.com/science/article/pii/S1574954126004760), [Hjort](https://doi.org/10.3389/frai.2026.1930016).

## Published recipe vs this implementation

The publication uses COCO-pretrained MMDetection detectors, flight-grouped splits, ten epochs, 500 warmup iterations, learning-rate drops at 6/8, horizontal/vertical flips and a 2650x1768 resize. Its original search tested eight learning rates, followed by best-of-five training. We screen three rates over three epochs, then train a fresh ten-epoch model. These are budgeted departures, not a claim to have repeated their complete experiment.

The paper's Table 2 calls GFL's backbone `FasterRCNN X101-32x6d-dcnv2`, which does not match a standard GFL configuration; it also gives 53.07M parameters. We use MMDetection's closest named `gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco` and its official COCO checkpoint, explicitly as an inference. The authors' exact configuration, learning rate selected per dataset, gradient clipping value and trained deer weights were not located. No precision beyond the accessible evidence is implied.

Implementation sources: [GFL original paper](https://arxiv.org/abs/2006.04388), [MMDetection 2.28.2 GFL model zoo](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/configs/gfl/README.md), [Sherbrooke dataset](https://doi.org/10.5683/SP3/TELIYD), [Jones decoys](https://scholarsjunction.msstate.edu/cfr-publications/24/), [AWIR](https://scholarsjunction.msstate.edu/gri-publications/2/).

DonaNet's current repository provides an Ultralytics YOLO training interface; it does not by itself establish an independent detector family: https://github.com/wildintelproject/donanet.

## Honest comparisons

Published numbers belong to different datasets and protocols; do not rank models by those raw numbers. Report locally measured model results separately from this literature table. Use our frozen decoy test subset for all team checkpoints, with validation-only threshold tuning. The test contains one physical field/decoy arrangement and cannot demonstrate new-site transfer. Annotation quality and small targets are material limitations.
