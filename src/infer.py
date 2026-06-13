from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from quality import list_images, read_image, eme_score, blur_score
from train_baseline import extract_features


def predict(model_path: str | Path, input_path: str | Path, output_csv: str | Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    rows = []
    for path in list_images(input_path):
        x = extract_features(path).reshape(1, -1)
        pred = model.predict(x)[0]
        conf = None
        if hasattr(model, "predict_proba"):
            conf = float(model.predict_proba(x).max())
        img = read_image(path)
        blur = blur_score(img)
        rows.append({
            "path": str(path),
            "prediction": pred,
            "confidence": conf,
            "eme": eme_score(img),
            "blur": blur,
            "review_needed": bool((conf is not None and conf < 0.70) or blur < 80),
        })
    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cat ear-tip inference on images.")
    parser.add_argument("input", help="Image or image folder")
    parser.add_argument("--model", default="models/cat_ear_baseline.joblib")
    parser.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()
    df = predict(args.model, args.input, args.output)
    print(f"Wrote {len(df)} predictions to {args.output}")
