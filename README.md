# Colony Cat Management

Machine-learning project for supporting Trap-Neuter-Return (TNR) colony operations with cat image analysis.

Initial focus: **cat ear-tip / ear-clip detection** to distinguish likely sterilized cats from cats needing manual review or trapping priority.

## Current MVP

This repo contains a starter ML pipeline for:

- scoring image quality with EME and blur metrics
- preprocessing images with contrast stretching and optional gamma correction
- training a baseline clipped vs unclipped classifier
- running inference with confidence and manual-review flags

## Dataset layout

Add labeled images locally using this structure:

```text
data/raw/
  clipped/
    cat_001.jpg
  unclipped/
    cat_002.jpg
  unknown/
    optional_unlabeled_images.jpg
```

Do not train ambiguous images as negative examples. Put unclear, blurry, occluded, cropped, or side-angle images in `unknown/` or a future `unclear/` class.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Score image quality

```bash
python src/quality.py data/raw --output outputs/image_quality.csv
```

## Preprocess images

```bash
python src/preprocess.py data/raw data/processed --gamma 1.2
```

## Train baseline classifier

```bash
python src/train_baseline.py --data data/raw --model-out models/cat_ear_baseline.joblib --report-out outputs/baseline_report.txt
```

## Run inference

```bash
python src/infer.py data/raw/unknown --model models/cat_ear_baseline.joblib --output outputs/predictions.csv
```

Output columns:

- `prediction`: clipped or unclipped
- `confidence`: model confidence when available
- `eme`: local contrast score
- `blur`: sharpness score
- `review_needed`: low-confidence or blurry image flag

## Recommended model path

This should not stay as a simple image classifier. Stronger architecture:

```text
Animal/cat detector
  -> cat head cropper
  -> left/right ear detector or segmentation
  -> ear status classifier
  -> cat-level clipped / unclipped / unclear decision
```

## Minimum dataset target

| Phase | Images needed |
|---|---:|
| Smoke test | 20 clipped + 20 unclipped |
| Baseline demo | 100 clipped + 100 unclipped |
| Usable prototype | 500+ per class |
| Field-ready | Thousands plus an `unclear` class |

## Ethical constraints

Use for animal welfare/TNR support. Avoid human surveillance from trail-camera captures. Do not use predictions as irreversible operational decisions without human review.
