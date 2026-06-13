from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from quality import list_images, read_image, contrast_stretch, gamma_correct


def preprocess(input_dir: str | Path, output_dir: str | Path, gamma: float | None = None) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    count = 0
    for path in list_images(input_dir):
        rel = path.relative_to(input_dir) if path.is_relative_to(input_dir) else Path(path.name)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = read_image(path)
        enhanced = contrast_stretch(img)
        if gamma is not None:
            enhanced = gamma_correct(enhanced, gamma)
        cv2.imwrite(str(out_path), enhanced)
        count += 1
    print(f"Processed {count} images into {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images with contrast stretching/gamma correction.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--gamma", type=float, default=None)
    args = parser.parse_args()
    preprocess(args.input_dir, args.output_dir, args.gamma)
