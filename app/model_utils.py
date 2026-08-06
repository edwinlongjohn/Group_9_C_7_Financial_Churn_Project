"""Model loading, validation, and prediction helpers for the churn app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cloudpickle
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "bank_churn_model_bundle.pkl"
SCHEMA_PATH = APP_DIR / "feature_schema.json"
METADATA_PATH = APP_DIR / "model_metadata.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_assets() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the trusted exported bundle and its supporting metadata."""
    with MODEL_PATH.open("rb") as file:
        bundle = cloudpickle.load(file)

    schema = read_json(SCHEMA_PATH)
    metadata = read_json(METADATA_PATH)

    required_keys = {
        "pipeline",
        "classification_threshold",
        "risk_tier_boundaries",
        "feature_schema",
        "metadata",
    }
    missing = required_keys.difference(bundle)
    if missing:
        raise ValueError(f"The model bundle is missing: {sorted(missing)}")

    expected_features = schema["feature_names_in_order"]
    if len(expected_features) != schema["number_of_features"]:
        raise ValueError("The feature count does not match the feature schema.")

    return bundle, schema, metadata


def prepare_input(
    values: dict[str, Any], schema: dict[str, Any]
) -> pd.DataFrame:
    """Create one correctly ordered, typed model-input row."""
    ordered = schema["feature_names_in_order"]
    missing = [name for name in ordered if name not in values]
    if missing:
        raise ValueError(f"Missing input fields: {', '.join(missing)}")

    frame = pd.DataFrame([{name: values[name] for name in ordered}])
    for name, dtype in schema["feature_dtypes"].items():
        if dtype == "int64":
            frame[name] = pd.to_numeric(frame[name]).astype("int64")
        elif dtype == "float64":
            frame[name] = pd.to_numeric(frame[name]).astype("float64")
        else:
            frame[name] = frame[name].astype("object")
    return frame


def assign_risk_tier(probability: float, boundaries: dict[str, float]) -> str:
    if probability < boundaries["low_upper"]:
        return "Low"
    if probability < boundaries["moderate_upper"]:
        return "Moderate"
    if probability < boundaries["high_upper"]:
        return "High"
    return "Very High"


def predict_customer(
    values: dict[str, Any],
    bundle: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Return probability, threshold decision, tier, and the typed input row."""
    frame = prepare_input(values, schema)
    probability = float(bundle["pipeline"].predict_proba(frame)[:, 1][0])
    threshold = float(bundle["classification_threshold"])
    predicted_class = int(probability >= threshold)
    tier = assign_risk_tier(probability, bundle["risk_tier_boundaries"])

    return {
        "churn_probability": probability,
        "predicted_class": predicted_class,
        "risk_tier": tier,
        "threshold": threshold,
        "input_frame": frame,
    }

