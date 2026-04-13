from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402


SAMPLE_FORM = {
    "school": "GP",
    "sex": "F",
    "age": "17",
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": "2",
    "Fedu": "2",
    "Mjob": "teacher",
    "Fjob": "services",
    "reason": "course",
    "nursery": "yes",
    "internet": "yes",
    "guardian_mat": "mother",
    "traveltime_mat": "1",
    "studytime_mat": "2",
    "failures_mat": "0",
    "schoolsup_mat": "no",
    "famsup_mat": "yes",
    "paid_mat": "no",
    "activities_mat": "yes",
    "higher_mat": "yes",
    "romantic_mat": "no",
    "famrel_mat": "4",
    "freetime_mat": "3",
    "goout_mat": "3",
    "Dalc_mat": "1",
    "Walc_mat": "1",
    "health_mat": "4",
    "absences_mat": "2",
    "guardian_por": "mother",
    "traveltime_por": "1",
    "studytime_por": "2",
    "failures_por": "0",
    "schoolsup_por": "no",
    "famsup_por": "yes",
    "paid_por": "no",
    "activities_por": "yes",
    "higher_por": "yes",
    "romantic_por": "no",
    "famrel_por": "4",
    "freetime_por": "3",
    "goout_por": "3",
    "Dalc_por": "1",
    "Walc_por": "1",
    "health_por": "4",
    "absences_por": "2",
}


def main() -> None:
    model_path = ROOT / "model" / "student_risk_model.pkl"
    encoders_path = ROOT / "model" / "label_encoders.pkl"
    feature_order_path = ROOT / "model" / "feature_order.pkl"

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    feature_order = joblib.load(feature_order_path)

    df = app.build_feature_row(SAMPLE_FORM)
    prediction = int(model.predict(df)[0])
    probabilities = model.predict_proba(df)[0].tolist() if hasattr(model, "predict_proba") else None

    leaf_probability_count = None
    if hasattr(model, "tree_") and model.tree_.value.shape[-1] > 1:
        leaf_mask = model.tree_.children_left == model.tree_.children_right
        leaf_values = model.tree_.value[leaf_mask][:, 0, :]
        leaf_probs = leaf_values / leaf_values.sum(axis=1, keepdims=True)
        leaf_probability_count = len({round(float(prob), 6) for prob in leaf_probs[:, 1]})

    report = {
        "model_type": type(model).__name__,
        "classes": [int(value) for value in getattr(model, "classes_", [])],
        "feature_count": len(feature_order),
        "encoder_count": len(encoders),
        "sample_row_shape": list(df.shape),
        "sample_prediction": prediction,
        "sample_probabilities": probabilities,
        "leaf_probability_count": leaf_probability_count,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
