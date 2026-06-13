# Dataset Notes

Images are intentionally not committed by default.

Use this local layout:

```text
data/raw/
  clipped/
  unclipped/
  unknown/
```

## Current seed labels from chat

- Unclipped seed set: about 20 images uploaded.
- Clipped seed set: about 19 images uploaded.

Use these for a smoke test only. The first usable prototype should target at least 500 images per class plus an `unclear` class.

## Labeling guidance

Good `clipped`:

- visible ear tip removed or notched
- ear is not hidden by angle, fur, object, or blur

Good `unclipped`:

- visible intact ear tips
- both ears or clearly intact target ear visible

Use `unknown` / future `unclear` for:

- obscured ears
- motion blur
- side or low-angle views where tip status is ambiguous
- heavy watermark/source artifacts
- multiple cats where target cat is unclear
- kittens or unusual ear morphology until enough examples exist
