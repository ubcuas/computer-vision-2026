# Competition Ambiguities and Open Questions

This file distinguishes unresolved competition-rule questions from engineering unknowns. Do not treat a working assumption as an official requirement.

## Organizer clarification candidates

| ID | Question | Why it matters | Current status |
| --- | --- | --- | --- |
| RULE-001 | Does a 10 m cluster use a fixed cluster center, or do deer form a connected cluster when each deer is within 10 m of another member? | Different definitions can produce different cluster counts from the same deer positions. | Open; no working definition should be finalized without clarification or an explicitly reviewed fallback. |
| RULE-002 | Clause 7 says “2-digit alphanumeric,” while clause 8 says “2-character capitalized alphanumeric.” Should all letters A-Z and digits 0-9 be expected in either position? | Determines the recognition vocabulary and validation rules. | Open; examples contain one digit and one capital letter, but examples may not define the full vocabulary. |
| RULE-003 | Will tag orientation, placement, material, and size match the examples apart from the guaranteed minimum font size and two-sided code? | Drives camera resolution, tag-detector training, and OCR augmentation. | Open. Only the properties stated in the CONOPS should be assumed. |
| RULE-004 | Is there a required machine-readable report schema beyond clearly labeled plain text and the required filename? | Affects automated report generation and validation. | Open; current document provides an example, not a formal schema. |

## Engineering unknowns

| ID | Question | How to resolve |
| --- | --- | --- |
| ENG-001 | What ground sampling distance and view angles will make a roughly 3 cm tag character reliably readable? | Camera and flight geometry calculation followed by controlled collection at representative altitudes and angles. |
| ENG-002 | Is a fixed camera sufficient, or is gimbal control needed for tag acquisition? | Compare coverage, resolution, motion blur, operational complexity, and available hardware in a representative test. |
| ENG-003 | How should repeated deer sightings across overlapping images be associated? | Compare geolocation, tracking, image stitching, and mapping approaches using a dataset with surveyed deer locations. |
| ENG-004 | Which available public datasets are legally usable and visually representative of plastic decoys viewed from the competition UAV? | Complete the dataset register, inspect licenses and distributions, and run comparable baselines. |
| ENG-005 | How should missing body parts, clothing or other artifacts, and severe occlusion be represented in training and evaluation? | Define controlled test slices and collect or synthesize representative examples without contaminating the final test set. |

## Resolution procedure

When a question is resolved:

1. Record the answer and evidence here.
2. State whether it is an official clarification, team decision, or experimental finding.
3. Link the preserved source, decision record, or experiment summary.
4. Update affected documentation and tests.
5. Keep the original question visible for historical context.
