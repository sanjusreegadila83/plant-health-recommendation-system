"""Image feature extraction and scikit-learn prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).resolve().parent / "models" / "plant_model.pkl"
FEATURE_VERSION = 1
IMAGE_SIZE = (64, 64)


class ModelNotFoundError(FileNotFoundError):
    """Raised when a trained model artifact is not present."""


def load_image(image_source: Any) -> Image.Image:
    """Open an image from a path, PIL image, or file-like object and return RGB."""
    if isinstance(image_source, Image.Image):
        image = image_source.copy()
    else:
        image = Image.open(image_source)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int] = IMAGE_SIZE) -> Image.Image:
    """Resize the image to the fixed feature-extraction dimensions."""
    return image.resize(size, Image.Resampling.LANCZOS)


def extract_features(image_source: Any) -> np.ndarray:
    """Extract normalized color histograms, spatial color, and texture features."""
    image = resize_image(load_image(image_source))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0

    features: list[float] = []
    for color_space in (rgb, hsv):
        for channel in range(3):
            histogram, _ = np.histogram(
                color_space[:, :, channel], bins=16, range=(0.0, 1.0), density=False
            )
            normalized_histogram = histogram.astype(np.float32)
            normalized_histogram /= max(float(normalized_histogram.sum()), 1.0)
            features.extend(normalized_histogram.tolist())
        features.extend(color_space.mean(axis=(0, 1)).tolist())
        features.extend(color_space.std(axis=(0, 1)).tolist())

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    spatial_gray = Image.fromarray((gray * 255).astype(np.uint8)).resize(
        (16, 16), Image.Resampling.LANCZOS
    )
    features.extend((np.asarray(spatial_gray, dtype=np.float32).ravel() / 255.0).tolist())

    # Mean and standard deviation of neighboring pixel differences provide simple texture cues.
    horizontal = np.diff(gray, axis=1)
    vertical = np.diff(gray, axis=0)
    features.extend(
        [
            float(np.mean(np.abs(horizontal))),
            float(np.std(horizontal)),
            float(np.mean(np.abs(vertical))),
            float(np.std(vertical)),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def _severity_for_label(label: str) -> str:
    normalized = label.casefold().replace("_", " ").replace("-", " ")
    if "unhealthy" not in normalized and any(
        term in normalized for term in ("healthy", "normal")
    ):
        return "Healthy"
    if any(
        term in normalized
        for term in ("severe", "advanced", "critical", "high risk", "late blight")
    ):
        return "Severe Disease"
    return "Leaf Disease"


def _assessment(severity: str) -> tuple[str, int, str]:
    if severity == "Healthy":
        return "Healthy", 90, "Low"
    if severity == "Severe Disease":
        return "Severe Disease", 30, "High"
    return "Leaf Disease", 65, "Moderate"


def predict_plant_health(
    image_source: Any, model_path: str | Path = MODEL_PATH
) -> dict[str, Any]:
    """Predict a class from a trained artifact; never substitutes a heuristic prediction."""
    artifact_path = Path(model_path)
    if not artifact_path.is_file():
        raise ModelNotFoundError(
            f"Trained model not found at {artifact_path}. Run `python train_model.py`."
        )
    try:
        artifact = joblib.load(artifact_path)
    except (EOFError, OSError, ValueError, TypeError, ImportError) as error:
        raise ValueError(
            "The saved model could not be loaded. Run `python train_model.py` to recreate it."
        ) from error
    if not isinstance(artifact, dict) or "classifier" not in artifact:
        raise ValueError("The model artifact has an unsupported format. Train it again.")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model feature version is incompatible. Run `python train_model.py`.")

    features = extract_features(image_source).reshape(1, -1)
    classifier = artifact["classifier"]
    predicted_label = str(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    confidence = float(np.max(probabilities))
    status = _severity_for_label(predicted_label)
    health_status, health_score, risk_level = _assessment(status)
    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "health_status": health_status,
        "health_score": health_score,
        "risk_level": risk_level,
    }
