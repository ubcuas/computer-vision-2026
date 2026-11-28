# Tag detection and tag-ID reading (prototype)

Status: exploratory, started 2026-11-28. No measured results yet; the first GPU run is pending.
Workstreams 4 and 5 in [Task 1](../../docs/competition/task-1.md). Tag IDs are worth 20 of the 40 raw
CV points, the largest single item.

## What the CONOPS guarantees (confirmed, CONOPS v1.0 §5.2.3 items 7-8, Figure 2)

- Some deer have a laminated paper tag on the ear; same code on both faces.
- Code = two capitalized alphanumeric characters, font 85 pt (~3 cm) or greater.
- Figure 2 shows white portrait cards with a punched hole and black Times-style serif text.

Everything else (font face, card size, how the tag hangs) is unconfirmed; see RULE-002 and RULE-003
in [ambiguities](../../docs/competition/ambiguities.md).

## Finding 1: camera geometry, not the model, is the main constraint (interpretation + calculation)

- 85 pt is the font's em size. Capitals and digits in a Times-style face are ~0.66 em, so the
  smallest guaranteed character is **~2.0 cm tall**, not 3 cm.
- An ear tag hangs roughly vertically, so a **nadir camera sees it nearly edge-on**. Reading it needs
  an oblique view (gimbal tilt or banked/low pass), and both faces carry the code, so either side works.
- `python geometry.py` gives character height in pixels. At a 45° oblique view:

| camera | 5 m | 10 m | 20 m | 30 m | max altitude for 14 px |
|---|---:|---:|---:|---:|---:|
| 1080p wide (~80°) | 2.3 px | 1.1 | 0.6 | 0.4 | ~1 m |
| 1080p normal (~50°) | 4.1 | 2.0 | 1.0 | 0.7 | ~2 m |
| 12 MP normal (~50°) | 8.6 | 4.3 | 2.2 | 1.4 | ~4 m |
| 1080p zoom (~15°) | 14.5 | 7.2 | 3.6 | 2.4 | ~7 m (slant) |
| 12 MP zoom (~15°) | 30.6 | 15.3 | 7.6 | 5.1 | ~15 m (slant) |

FOVs are illustrative; plug in the real camera with `--width-px --hfov`. The 14 px "reliable"
threshold is a placeholder until `evaluate.py` measures it. **Implication:** the 1080p survey camera
that the deer detector assumes cannot read tags from any practical survey altitude. Tag reading
needs a separate acquisition step: a narrow-FOV/zoom (ideally high-MP) camera on a gimbal, or
low oblique passes over each detected deer. That is a hardware and flight-plan question for the
airframe and payload teams, and it bears on ENG-001 and ENG-002.

## Proposed pipeline (hypothesis, for team review)

1. **Survey pass**: the existing deer detector + geolocation builds the deer inventory.
2. **Tag acquisition**: revisit each geolocated deer with an oblique, zoomed view (gimbal pointing at
   the deer's GPS position).
3. **Tag detection**: a small YOLO tag class run on the deer crop (expanded around the head), not the
   full frame. A white rectangle on a brown decoy is high contrast, so this should be the easy stage.
   Trade-off: if the deer detector misses a deer, its tag is missed too. A full-frame tiled tag
   detector is the fallback if the team wants tags independent of deer (task-1.md workstream 4 wording).
4. **Tag-ID recognition**: `TagNet`, two 36-way classifier heads on the tag crop (this folder).
5. **Multi-frame fusion**: one deer gives many crops. Sum per-character log-probabilities across
   sightings of the same deer and report the argmax, which beats any single frame. Each code is
   unique, so two deer resolving to the same ID flags a conflict for operator review.

Design choices so far:

- **Closed-vocabulary classifier vs general OCR (EasyOCR/PaddleOCR/Tesseract).** The ID is exactly 2
  chars from 36 symbols, so a classifier has no detection or sequence decoding to fail at tiny sizes,
  is ~1.2M params (cheap on the Jetson next to YOLO), and outputs per-character probabilities that
  multi-frame voting needs. General OCR is still worth one baseline run on real crops for comparison.
- **Synthetic training data first.** No public dataset has these tags, and the CONOPS specifies their
  appearance tightly. Training mixes serif and sans fonts so the model does not overfit to Times;
  unseen fonts are held out as an eval slice. Real photos of printed tags are still required before
  trusting any number (see next steps).
- **Pixel height as the evaluation axis.** It converts directly to camera range via `geometry.py`,
  which is the decision the team needs to make.

## Files

| file | purpose |
|---|---|
| `geometry.py` | character px vs camera, altitude, view angle |
| `synth.py` | renders one synthetic tag crop (card, font, 3D warp, glare, blur, noise, JPEG); `--sheet` previews |
| `data.py` | parallel generation + `.npz` cache (torch-free; one OpenCV thread per worker) |
| `recognizer.py` | `TagNet`, preprocessing, `predict()` returning ID + confidence |
| `train.py` | trains on synthetic data, CUDA if available |
| `evaluate.py` | exact-ID accuracy vs char px on `times`, `heldout`, `easy` slices + confusions |
| `Tag_OCR_GPU.ipynb` | runs all of the above on a GPU box and saves results for git |

## Running on a GPU machine

Easiest: pull the branch on the GPU machine, open `Tag_OCR_GPU.ipynb` from this folder and *Run All*.
It checks the GPU/fonts, trains, evaluates, plots, and copies the small results into
`results/<RUN_NAME>/`. Commit the notebook with its outputs plus that folder; never the weights.
Set `SMOKE = True` first for a 5-minute "does it run" check. The equivalent command-line steps:

```bash
pip install torch numpy opencv-python pillow           # torch with CUDA for your GPU
# Linux only: fonts (Liberation Serif stands in for Times New Roman)
sudo apt install fonts-liberation fonts-dejavu fonts-freefont-ttf
cd prototyping/tag-ocr
python synth.py --out ../../artifacts/tag-ocr/preview --n 48 --sheet   # eyeball the data
python train.py --out ../../artifacts/tag-ocr/run1                       # 200k samples, 30 epochs
python evaluate.py --model ../../artifacts/tag-ocr/run1/tagnet.pt         # writes eval.md / eval.csv
```

Rendering is CPU-bound (set `--workers`); training holds the whole uint8 set on the GPU (~1.2 GB for
200k). Outputs go to the git-ignored `artifacts/`. Smoke-tested only on CPU at tiny sizes (2k samples,
1 epoch); the scripts ran end to end, but no accuracy number exists yet.

## Next steps

1. Run `train.py` + `evaluate.py` on GPU; record the result as an experiment summary and replace the
   14 px placeholder in `geometry.py` with the measured threshold.
2. **Print real tags** (Times, 85 pt, laminated, ~5 x 8 cm card), hang them on a decoy or anything
   ear-height, and photograph at measured distances/angles with the actual flight camera. Label the
   crops. This is the first real test set and must stay out of training.
3. Get the real camera specs (sensor px, HFOV, zoom?) and gimbal availability; rerun `geometry.py`.
4. Train a YOLO tag-box detector on synthetic tags composited onto deer crops (Deer v3 decoy images),
   then validate on the real photos from step 2.
5. Implement multi-frame fusion once deer tracking/geolocation produces per-deer crop sets.
6. Ask organizers RULE-002/003 (vocabulary, ambiguous pairs like 0/O and 1/I, font, tag size).
