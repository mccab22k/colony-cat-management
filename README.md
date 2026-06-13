# Colony Cat Management

Using computer vision and machine learning to support Trap-Neuter-Return (TNR) programs, monitor colony health, and improve the welfare of community cats.

---

## Background

This project combines work from two different stages of my academic career.

As an undergraduate studying Computer Science, I worked on image processing and object identification problems involving high-speed particle tracking from video. Those projects required extracting useful information from noisy imagery, identifying edges and features, selecting high-quality frames, and tracking objects across time.

Later, in graduate school, I expanded into machine learning and statistical modeling, focusing on how models can identify meaningful patterns in imperfect real-world datasets.

This repository represents an effort to bring those experiences together and apply them to a problem that has both ecological and animal welfare impacts: monitoring and managing community cat populations.

---

## The Problem

Feral and community cats exist in cities, suburbs, and rural environments around the world. When unmanaged, populations can grow rapidly, creating challenges for:

* Cat health and welfare
* Disease transmission
* Resource competition
* Local wildlife populations
* Volunteer and shelter resources

The most humane and widely accepted population management strategy is Trap-Neuter-Return (TNR).

Under TNR programs, cats are:

1. Humanely trapped
2. Spayed or neutered
3. Vaccinated when possible
4. Returned to their colony

A small portion of one ear is removed during the procedure, creating a visible "ear tip" that serves as a permanent marker indicating the cat has already been sterilized.

---

## Why Ear Tipping Matters

Ear tipping is more than a simple identifier.

The percentage of cats within a colony that have been ear tipped can provide insight into the overall status of that colony.

A colony with a high percentage of tipped cats may indicate:

* Strong TNR coverage
* Reduced reproduction
* More stable population growth
* Lower demand on volunteer resources

A colony with a low percentage of tipped cats may indicate:

* New arrivals
* Population expansion
* Incomplete TNR coverage
* Areas where intervention may be beneficial

By monitoring ear-tip prevalence over time, organizations can estimate whether a colony is becoming more stable or whether additional trapping efforts may be required.

## Why TNR Percentage Matters

One of the strongest indicators of colony stability is the percentage of cats that have already been sterilized through Trap-Neuter-Return (TNR) programs.

Research and field experience have shown that cat populations generally begin to stabilize when a sufficiently large percentage of the colony has been sterilized. While exact thresholds vary between environments, many TNR programs target sterilization rates of 70–80% or higher.

A colony with a high proportion of ear-tipped cats often suggests:

* Effective TNR coverage
* Reduced kitten births
* Slower population growth
* Lower shelter intake pressure
* Improved allocation of volunteer resources

Conversely, a colony with a low proportion of ear-tipped cats may indicate:

* Active reproduction
* New arrivals from surrounding areas
* Incomplete TNR coverage
* Increased future trapping requirements

Because ear tipping serves as a visible marker of sterilization status, automated monitoring of tipped versus untipped cats can provide a practical proxy for colony health.

Over time, repeated observations can help answer questions such as:

* Is the colony growing or stabilizing?
* Are new unsterilized cats entering the population?
* Are TNR efforts achieving sufficient coverage?
* Which colonies should be prioritized for future interventions?

Rather than relying solely on manual surveys, computer vision systems may eventually provide a scalable way to estimate sterilization coverage and identify colonies that would benefit most from additional TNR efforts.


---

## Project Goals

### Phase 1: Ear Tip Detection

Determine whether a cat appears to be:

* Ear tipped
* Not ear tipped
* Uncertain / requires review

This serves as the foundation for automated TNR monitoring.

### Phase 2: Automated Colony Monitoring

Process images from:

* Trail cameras
* Wildlife cameras
* Security cameras
* Volunteer photographs
* Mobile devices

to estimate:

* Number of cats present
* Percentage tipped
* New arrivals
* Temporal population trends

### Phase 3: Colony Health Analytics

Use longitudinal observations to estimate:

* TNR saturation rates
* Population stability
* Colony growth trends
* Potential intervention priorities

---

## Technical Approach

This project combines classical image processing techniques with modern machine learning.

### Image Quality Assessment

Many field images are:

* Blurry
* Poorly lit
* Low contrast
* Motion affected

Image quality metrics such as Enhancement Measure Estimation (EME) and blur detection can help identify the most useful frames before analysis.

### Image Enhancement

Preprocessing techniques include:

* Contrast stretching
* Gamma correction
* Adaptive histogram methods
* Future denoising pipelines

### Machine Learning

Current development focuses on:

* Ear-tip classification
* Feature extraction
* Traditional machine learning baselines

Future versions will incorporate:

* Convolutional Neural Networks (CNNs)
* YOLO object detection
* Ear segmentation
* Cat re-identification
* Temporal tracking across video sequences

---

## Long-Term Vision

The long-term goal is not simply to determine whether a single cat has a clipped ear.

The goal is to build tools that help volunteers and animal welfare organizations better understand colony dynamics while minimizing manual effort.

A future workflow may look like:

Wildlife Camera
→ Animal Detection
→ Cat Detection
→ Ear Identification
→ TNR Status Classification
→ Colony Statistics Dashboard

Providing:

* Estimated colony size
* Tipped vs untipped percentages
* Potential newcomers
* Historical trends
* Colony health indicators

---

## Research Motivation

This project sits at the intersection of:

* Computer Vision
* Machine Learning
* Ecology
* Animal Welfare
* Citizen Science

It represents an opportunity to apply techniques originally developed for scientific image analysis and machine learning research to a practical problem with direct community impact.

By improving our ability to monitor colony populations, we can help direct limited volunteer resources toward the cats and colonies that need them most.

---

## Status

Early development.

Current work focuses on:

* Building labeled ear-tip datasets
* Evaluating image quality metrics
* Developing baseline classifiers
* Preparing for object-detection and segmentation models

Contributions, feedback, and collaboration from TNR organizations, rescuers, researchers, and volunteers are welcome.
--------------

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
