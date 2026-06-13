from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def read_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def eme_score(img: np.ndarray, block_size: int = 10, eps: float = 1e-6) -> float:
    """Enhancement Measure Estimation. Higher generally means stronger local contrast."""
    gray = to_gray(img).astype(np.float32)
    h, w = gray.shape
    total = 0.0
    blocks = 0
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            bmin = float(block.min()) + eps
            bmax = float(block.max()) + eps
            total += 20.0 * math.log(bmax / bmin)
            blocks += 1
    return total / max(blocks, 1)


def blur_score(img: np.ndarray) -> float:
    """Variance of Laplacian. Higher generally means sharper image."""
    gray = to_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def contrast_stretch(img: np.ndarray, low_pct: float = 2, high_pct: float = 98) -> np.ndarray:
    """Percentile contrast stretch, safer than fixed r1/s1/r2/s2 constants."""
    arr = img.astype(np.float32)
    lo, hi = np.percentile(arr, (low_pct, high_pct))
    if hi <= lo:
        return img.copy()
    out = (arr - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_correct(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def score_images(input_dir: str | Path, output_csv: str | Path) -> pd.DataFrame:
    rows = []
    for path in list_images(input_dir):
        img = read_image(path)
        rows.append({
            "path": str(path),
            "file": path.name,
            "label": path.parent.name,
            "eme_original": eme_score(img),
            "blur_original": blur_score(img),
            "eme_contrast": eme_score(contrast_stretch(img)),
            "eme_gamma_0_8": eme_score(gamma_correct(img, 0.8)),
            "eme_gamma_1_2": eme_score(gamma_correct(img, 1.2)),
        })
    df = pd.DataFrame(rows).sort_values(["label", "eme_original"], ascending=[True, False]) if rows else pd.DataFrame()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score image quality for cat ear-tip detection.")
    parser.add_argument("input_dir", help="Folder containing images, optionally in class subfolders")
    parser.add_argument("--output", default="outputs/image_quality.csv")
    args = parser.parse_args()
    df = score_images(args.input_dir, args.output)
    print(f"Wrote {len(df)} rows to {args.output}")
