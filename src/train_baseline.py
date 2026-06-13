from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quality import IMAGE_EXTS, read_image, to_gray, contrast_stretch, eme_score, blur_score

VALID_LABELS = {"clipped", "unclipped"}


def load_labeled_images(data_dir: str | Path) -> list[tuple[Path, str]]:
    data_dir = Path(data_dir)
    rows: list[tuple[Path, str]] = []
    for label in sorted(VALID_LABELS):
        for p in sorted((data_dir / label).rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append((p, label))
    if not rows:
        raise ValueError(f"No labeled images found. Expected: {data_dir}/clipped and {data_dir}/unclipped")
    labels = {label for _, label in rows}
    missing = VALID_LABELS - labels
    if missing:
        raise ValueError(f"Missing class folders/images for: {sorted(missing)}")
    return rows


def extract_features(path: Path, size: int = 160) -> np.ndarray:
    img = read_image(path)
    img = contrast_stretch(img)
    gray = to_gray(img)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10), density=True)

    stats = np.array([
        gray.mean(), gray.std(), gray.min(), gray.max(),
        eme_score(gray), blur_score(gray),
    ], dtype=np.float32)
    return np.concatenate([hog_features.astype(np.float32), lbp_hist.astype(np.float32), stats])


def train(data_dir: str | Path, model_out: str | Path, report_out: str | Path) -> None:
    rows = load_labeled_images(data_dir)
    X = np.vstack([extract_features(p) for p, _ in rows])
    y = np.array([label for _, label in rows])

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    report_lines = []
    class_counts = pd.Series(y).value_counts().to_dict()
    report_lines.append(f"Class counts: {class_counts}\n")

    if min(class_counts.values()) >= 3 and len(y) >= 8:
        n_splits = min(5, min(class_counts.values()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
        report_lines.append(f"CV balanced accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}, folds={scores.tolist()}\n")

    if min(class_counts.values()) >= 2 and len(y) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report_lines.append("Holdout classification report:\n")
        report_lines.append(classification_report(y_test, preds, zero_division=0))
        report_lines.append("Confusion matrix:\n")
        report_lines.append(str(confusion_matrix(y_test, preds, labels=sorted(VALID_LABELS))))
        report_lines.append("\n")

    model.fit(X, y)
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": sorted(VALID_LABELS), "feature_size": X.shape[1]}, model_out)

    report_out = Path(report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved model: {model_out}")
    print(f"Saved report: {report_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline clipped/unclipped cat-ear classifier.")
    parser.add_argument("--data", default="data/raw", help="Folder with clipped/ and unclipped/ subfolders")
    parser.add_argument("--model-out", default="models/cat_ear_baseline.joblib")
    parser.add_argument("--report-out", default="outputs/baseline_report.txt")
    args = parser.parse_args()
    train(args.data, args.model_out, args.report_out)
