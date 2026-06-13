# Model Card: Baseline Cat Ear-Tip Classifier

## Intended use

Research/demo classifier for TNR triage support. Not a replacement for human verification.

## Inputs

Cat images, ideally cropped to head/ears.

## Outputs

- clipped
- unclipped
- confidence
- review_needed flag

## Limitations

- Not trained without user-provided labeled images.
- Poor on full-body images unless ears are visible.
- Poor on motion blur, occlusion, low light, unusual poses, kittens, folded-ear breeds.
- Requires balanced clipped/unclipped dataset.
- Watermarked stock photos should not be used in a public training dataset unless licensing allows it.

## Bias and failure modes

- Many TNR photos show left-ear tipping. Training must use horizontal flips and mixed examples so the model learns tipped-ear geometry rather than side bias.
- A model may learn background/source artifacts if clipped and unclipped photos come from different websites or camera types.
- Ambiguous images should be labeled `unclear`, not forced into `clipped` or `unclipped`.

## Ethical constraints

- Use for animal welfare/TNR support.
- Avoid human surveillance from trail-camera captures.
- Avoid irreversible operational decisions without manual review.
