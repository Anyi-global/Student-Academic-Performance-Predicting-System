from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


MERGE_KEYS = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
]

DROP_COLUMNS = ["G1_mat", "G2_mat", "G3_mat", "G1_por", "G2_por", "G3_por", "G3_avg"]

MODEL_PARAMS = {
    "class_weight": "balanced",
    "random_state": 42,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "ccp_alpha": 0.0,
}


@dataclass
class HoldoutMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    unique_probability_count: int
    leaf_count: int


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Retrain the student risk model safely.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "model" / "student-performance-data" / "dataset",
        help="Directory containing student-mat.csv and student-por.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "model",
        help="Directory where model artifacts should be written.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=root / "model" / "backups",
        help="Directory used to store backups of existing artifacts.",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Write artifacts without first backing up the existing ones.",
    )
    return parser.parse_args()


def load_and_prepare_dataset(data_dir: Path) -> pd.DataFrame:
    mat = pd.read_csv(data_dir / "student-mat.csv", sep=";")
    por = pd.read_csv(data_dir / "student-por.csv", sep=";")

    merged = pd.merge(mat, por, on=MERGE_KEYS, suffixes=("_mat", "_por"))
    merged["G3_avg"] = (merged["G3_mat"] + merged["G3_por"]) / 2
    merged["risk"] = (merged["G3_avg"] < 10).astype(int)
    merged = merged.drop(columns=DROP_COLUMNS)

    # Keep feature engineering aligned with the deployed Flask app contract.
    merged["parent_edu_avg"] = (merged["Medu"] + merged["Fedu"]) / 2
    merged["lifestyle_score"] = merged["goout_mat"] + merged["freetime_mat"]
    return merged


def fit_label_encoders(df: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    for column in df.columns:
        if column == "risk":
            continue
        if is_object_dtype(df[column]) or is_string_dtype(df[column]):
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            encoders[column] = encoder
    return encoders


def evaluate_model(X: pd.DataFrame, y: pd.Series, params: dict) -> HoldoutMetrics:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    unique_probability_count = len({round(float(value), 6) for value in probabilities})

    return HoldoutMetrics(
        accuracy=round(float(accuracy_score(y_test, predictions)), 4),
        precision=round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        recall=round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        f1=round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_test, probabilities)), 4),
        unique_probability_count=unique_probability_count,
        leaf_count=int(model.tree_.n_leaves),
    )


def cross_validate_model(X: pd.DataFrame, y: pd.Series, params: dict) -> dict[str, float]:
    model = DecisionTreeClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
    )
    return {
        key.replace("test_", ""): round(float(value.mean()), 4)
        for key, value in scores.items()
        if key.startswith("test_")
    }


def backup_existing_artifacts(output_dir: Path, backup_root: Path) -> Path | None:
    artifact_names = ["student_risk_model.pkl", "label_encoders.pkl", "feature_order.pkl", "training_report.json"]
    existing = [output_dir / name for name in artifact_names if (output_dir / name).exists()]
    if not existing:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = backup_root / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for artifact in existing:
        shutil.copy2(artifact, backup_dir / artifact.name)
    return backup_dir


def save_artifacts(
    output_dir: Path,
    model: DecisionTreeClassifier,
    encoders: dict[str, LabelEncoder],
    feature_order: list[str],
    training_report: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "student_risk_model.pkl")
    joblib.dump(encoders, output_dir / "label_encoders.pkl")
    joblib.dump(feature_order, output_dir / "feature_order.pkl")
    (output_dir / "training_report.json").write_text(json.dumps(training_report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prepared = load_and_prepare_dataset(args.data_dir)
    encoders = fit_label_encoders(prepared)

    X = prepared.drop(columns=["risk"])
    y = prepared["risk"]
    feature_order = X.columns.tolist()

    baseline_params = {"class_weight": "balanced", "random_state": 42}
    baseline_holdout = evaluate_model(X, y, baseline_params)
    candidate_holdout = evaluate_model(X, y, MODEL_PARAMS)
    baseline_cv = cross_validate_model(X, y, baseline_params)
    candidate_cv = cross_validate_model(X, y, MODEL_PARAMS)

    final_model = DecisionTreeClassifier(**MODEL_PARAMS)
    final_model.fit(X, y)

    backup_dir = None
    if not args.skip_backup:
        backup_dir = backup_existing_artifacts(args.output_dir, args.backup_dir)

    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "backup_dir": str(backup_dir) if backup_dir else None,
        "feature_count": len(feature_order),
        "encoder_count": len(encoders),
        "selected_model_params": MODEL_PARAMS,
        "baseline_holdout": asdict(baseline_holdout),
        "candidate_holdout": asdict(candidate_holdout),
        "baseline_cross_validation": baseline_cv,
        "candidate_cross_validation": candidate_cv,
    }

    save_artifacts(args.output_dir, final_model, encoders, feature_order, report)

    print("Saved artifacts to:", args.output_dir)
    if backup_dir:
        print("Backed up existing artifacts to:", backup_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
