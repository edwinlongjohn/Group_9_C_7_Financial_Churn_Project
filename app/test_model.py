"""Round-trip verification using the sample packaged with the exported model."""

import json
from pathlib import Path

import pandas as pd

from model_utils import load_assets, predict_customer


app_dir = Path(__file__).resolve().parent
bundle, schema, _ = load_assets()
sample = pd.read_csv(app_dir / "sample_customer_input.csv").iloc[0].to_dict()
expected = json.loads((app_dir / "sample_expected_output.json").read_text())
actual = predict_customer(sample, bundle, schema)

assert abs(actual["churn_probability"] - expected["churn_probability"]) < 1e-12
assert actual["predicted_class"] == expected["predicted_class"]
assert actual["risk_tier"] == expected["risk_tier"]

print("PASS: exported model reproduces the expected sample prediction exactly.")
print(f"Probability: {actual['churn_probability']:.15f}")
print(f"Class: {actual['predicted_class']}")
print(f"Risk tier: {actual['risk_tier']}")

