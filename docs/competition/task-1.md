# Task 1: Herd Survey

- Status: working team summary
- Rules baseline: 2027 AEAC CONOPS v1.0, dated 2026-09-08, section 5.2.3 and Table 5
- Authority: the official CONOPS and written organizer clarifications take precedence over this summary

## Objective

Survey the defined search area with one UAV and report every deer decoy, the deer count per spatial cluster, visible ear-tag IDs, and anomalous deer interactions with human artifacts.

## Confirmed competition requirements

### Scene and targets

- Targets are plastic deer decoys; the exact model may vary.
- Deer may have antlers or no antlers.
- Some deer have laminated paper ear tags.
- A tag has the same unique code on both faces.
- The tag code is two capitalized alphanumeric characters. The CONOPS describes the font as 85 pt, approximately 3 cm, or greater.

### Required report contents

- Total number of deer in the search area.
- Number of deer in each cluster, described by the CONOPS as having a 10 m radius per cluster.
- A deer alone within a 10 m radius is a cluster of one.
- The two-character capitalized alphanumeric ID from every detected tag.
- A brief plain-language description of deer interacting anomalously with human artifacts, for example a deer wearing a hat.

### Delivery

- Submit a plain-text file named `<your_team_name>_task1_survey.txt` to the team's Google Drive folder under `Phase 2 Deliverables`.
- Clearly label every required piece of information.
- Upload no later than 15 minutes after the end of the flight window.
- The deer-detection and description score is multiplied according to submission time: 150% at five minutes before the end of the flight window or earlier, 100% at 15 minutes after the window, linearly interpolated between those points, and 0% after that deadline.

### Operational dependencies

- Only one UAV is permitted and its battery may not be swapped during the task.
- The prescribed course to the search area is between 3 km and 5 km.
- The UAS must transmit telemetry while armed and participate in simulated airspace deconfliction.
- The competition server requires 1 Hz telemetry. Full telemetry and flight requirements remain owned by the official CONOPS and the relevant software subteams.

## Overall Task 1 scoring context

| Criterion | Maximum points |
| --- | ---: |
| Flight speed to the survey area | 30 |
| Simulated airspace deconfliction | 25 |
| Deer detection and description, including the report-speed multiplier | 60 |
| Safe landing at the flight line | 5 |
| **Total** | **120** |

The flight-speed and deconfliction details are important system constraints but are not owned solely by the computer-vision subteam. Refer to the official CONOPS for the complete definitions and penalties.

## Scoring relevant to computer vision

The deer detection and description portion is nominally worth 40 points before the report-speed multiplier:

- correct total deer count: 5 points;
- correct deer counts per 10 m cluster: 10 points total;
- correct tag IDs: 20 points total; and
- anomalous deer descriptions: 5 points.

This weighting makes tag recognition the largest direct CV scoring component, while end-to-end report latency can substantially change the achieved score.

## Current computer-vision workstreams

These are team engineering decompositions, not competition requirements:

1. **Deer detection:** locate antlered and non-antlered decoys across expected scales, viewpoints, backgrounds, damage, and partial occlusion.
2. **Persistent inventory:** associate detections across overlapping frames so repeated sightings do not inflate the total.
3. **Geolocation and clustering:** estimate deer positions and apply the eventual agreed interpretation of the 10 m cluster rule.
4. **Tag detection:** localize small laminated tags independently of the deer detector.
5. **Tag text recognition:** recover two-character IDs using sufficiently high-resolution imagery and dedicated recognition or OCR.
6. **Anomaly description:** identify interactions with human artifacts and produce a short human-readable description.
7. **Report generation:** convert verified results into the required text format early enough to preserve the submission multiplier.

## Near-term team outcomes

The initial team notes identify these concrete outcomes:

- identify an initial training dataset for deer detection;
- identify suitable data or a collection strategy for tag detection and tag-text recognition;
- investigate initial model and pipeline options for each workstream;
- define an evaluation plan that approximates competition conditions; and
- reserve a final locally collected dataset for an end-to-end test that is as close to competition conditions as practical.

These outcomes do not require a separate GitHub Issue before research begins. Substantial findings should be preserved in the dataset register, an experiment summary, or a technical decision as appropriate.

## Evaluation direction

The following are proposed evaluation targets and must be refined as the pipeline and data become clearer:

- Evaluate deer detection across altitude, scale, viewpoint, lighting, antler state, decoy type, occlusion, and background.
- Separate per-frame detection quality from unique-deer counting quality.
- Measure total-count and per-cluster exactness in addition to conventional detector metrics.
- Evaluate tag localization separately from exact two-character recognition.
- Include untagged deer so the pipeline cannot depend on tag presence.
- Reserve a final locally collected test set that approximates competition conditions and is not used for training or model selection.
- Run an end-to-end timed test from image capture through final report creation.

Exact metrics, thresholds, and dataset splits are unresolved until initial baselines and representative data exist.

## Research leads from initial team notes

These items are unverified leads, not endorsed datasets or model choices. Record URLs, licenses, data quality, and reproduction results before adopting them.

### Model and approach leads

- YOLO-based deer detection, followed by spatial grouping rather than treating a cluster as a visual class.
- A two-stage animal detector followed by deer classification.
- Dedicated tag detector followed by OCR or character recognition.
- Image stitching or 3D mapping for persistent inventory and geolocation.
- Research literature on animal counting from UAS imagery.
- Donanet, described in the notes as a YOLO-based deer identification, classification, and counting model; the notes indicate that claimed weights were not found.
- RucervusDAuceveli, described in the notes as a trained deer-detection model.

### Dataset leads

- BAMBI: multimodal nadir UAV recordings of forest wildlife.
- Mississippi State deer-decoy dataset: noted as 264 images and as previously used by the team, but the recorded positive and negative counts do not sum to 264; provenance, exact counts, split, and license still require verification.
- BorealisV2: birds-eye imagery that may contain a small number of deer; annotation suitability requires verification.

Move verified dataset details into `docs/datasets/README.md` rather than expanding this list indefinitely.

## Open questions

See `ambiguities.md`. Do not silently resolve competition-rule ambiguities in code or experiment assumptions.
