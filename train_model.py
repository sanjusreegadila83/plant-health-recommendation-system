"""Train and evaluate the local plant image classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB

from model import FEATURE_VERSION, extract_features

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_PATH = PROJECT_DIR / "models" / "plant_model.pkl"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read class-folder images and extract their feature vectors."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}. Create class subfolders and add images."
        )
    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {dataset_dir}.")

    features: list[np.ndarray] = []
    labels: list[str] = []
    skipped: list[str] = []
    for class_dir in class_dirs:
        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for image_path in image_paths:
            try:
                features.append(extract_features(image_path))
                labels.append(class_dir.name)
            except (OSError, ValueError) as error:
                skipped.append(f"{image_path}: {error}")

    if skipped:
        print(f"Skipped {len(skipped)} unreadable image(s).")
        for message in skipped[:5]:
            print(f"  - {message}")
    if not features:
        raise ValueError(
            f"No readable JPG/JPEG/PNG images found in {dataset_dir}. "
            "Add images inside one or more class folders."
        )
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError(
            "At least two classes with readable images are required to train a classifier."
        )
    class_counts = {label: labels.count(label) for label in classes}
    if any(count < 2 for count in class_counts.values()):
        low_count = ", ".join(f"{label}: {count}" for label, count in class_counts.items() if count < 2)
        raise ValueError(
            "Each class needs at least two readable images for training and evaluation. "
            f"Insufficient counts: {low_count}"
        )
    return np.vstack(features), np.asarray(labels), classes


def stratified_split(
    labels: np.ndarray, classes: list[str], test_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible train/test indices while retaining each class in both sets."""
    rng = np.random.default_rng(42)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in classes:
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        test_count = min(
            max(1, int(round(len(class_indices) * test_fraction))),
            len(class_indices) - 1,
        )
        test_indices.extend(class_indices[:test_count].tolist())
        train_indices.extend(class_indices[test_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.asarray(train_indices), np.asarray(test_indices)


def print_evaluation(
    truth: np.ndarray, predictions: np.ndarray, classes: list[str]
) -> float:
    """Print accuracy and per-class precision, recall, and F1 without optional native metrics."""
    accuracy = float(np.mean(truth == predictions))
    print(f"Held-out accuracy: {accuracy:.3f}")
    print(f"{'class':24} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>9}")
    for label in classes:
        true_positive = int(np.sum((truth == label) & (predictions == label)))
        false_positive = int(np.sum((truth != label) & (predictions == label)))
        false_negative = int(np.sum((truth == label) & (predictions != label)))
        support = true_positive + false_negative
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        print(
            f"{label:24} {precision:10.3f} {recall:10.3f} "
            f"{f1:10.3f} {support:9d}"
        )
    return accuracy


def train() -> None:
    """Train a Gaussian Naive Bayes classifier and save its evaluated artifact."""
    X, y, classes = load_dataset(DATASET_DIR)
    train_indices, test_indices = stratified_split(y, classes)
    classifier = GaussianNB()
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    print(f"Images: {len(y)} | Classes: {', '.join(classes)}")
    print_evaluation(y_test, predictions, classes)

    # Refit on every labeled image after evaluation so the saved demo model uses all available data.
    classifier.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "classes": classes,
            "feature_version": FEATURE_VERSION,
            "image_size": (64, 64),
        },
        MODEL_PATH,
    )
    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    try:
        train()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Training stopped: {error}") from error
