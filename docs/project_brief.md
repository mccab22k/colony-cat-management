# Project Brief: Cat Ear-Tip Identification for TNR

## Problem

Trap-Neuter-Return programs use ear tipping/clipping to mark cats that have already been sterilized. Camera-trap or phone images could help volunteers triage colonies, reduce duplicate trapping, and prioritize unaltered cats.

## MVP

Classify cat images as:

- clipped/tipped
- unclipped
- unclear/manual review

## Why this is hard

The signal is small: the model must inspect ear geometry, not just determine whether a cat is present. Failure modes include side angle, folded ears, dark fur, occlusion, poor lighting, motion blur, and low-resolution trail cameras.

## Technical approach

1. Select best image/frame using EME and blur scoring.
2. Normalize/enhance image with contrast stretching and optional gamma correction.
3. Train a classical baseline using HOG/texture features.
4. Move to detector + classifier once enough labeled data exists.
5. Preserve human review for low-confidence predictions.

## Evaluation

Primary metrics:

- balanced accuracy
- precision/recall for clipped class
- false-negative rate for clipped cats
- unclear/manual-review rate

Operationally, false confidence is worse than an `unclear` result.

## Future system

```text
Wildlife camera / phone image
  -> animal detected
  -> cat detected
  -> cat head/ear crop
  -> tipped / not tipped / unclear
  -> colony dashboard
```

Future colony dashboard features:

- estimated colony count
- re-sighting frequency
- suspected newcomers
- tipped/not tipped status
- injury or poor body-condition review queue
