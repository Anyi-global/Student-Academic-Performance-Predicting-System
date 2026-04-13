"""
Student Academic Risk Prediction System.
Flask web application for predicting student academic risk.
"""

import json
import os
import sqlite3
import traceback
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORY_DB_PATH = os.path.join(DATA_DIR, "prediction_history.db")


model = None
label_encoders = {}
feature_order = []
load_error = None
history_store_error = None


try:
    model = joblib.load(os.path.join(MODEL_DIR, "student_risk_model.pkl"))
    loaded_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    label_encoders = loaded_encoders if isinstance(loaded_encoders, dict) else {}
    feature_order = joblib.load(os.path.join(MODEL_DIR, "feature_order.pkl"))
    print(f"[OK] Model loaded - features: {len(feature_order)}, encoders: {len(label_encoders)}")
except Exception as exc:
    load_error = str(exc)
    print(f"[ERROR] Could not load model artifacts: {exc}")


# Manual fallback maps remain available if an encoder is missing for a field.
ENCODE = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"GT3": 0, "LE3": 1},
    "Pstatus": {"A": 0, "T": 1},
    "Mjob": {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
    "Fjob": {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
    "reason": {"course": 0, "home": 1, "other": 2, "reputation": 3},
    "guardian": {"father": 0, "mother": 1, "other": 2},
    "yesno": {"no": 0, "yes": 1},
}

YESNO_FIELDS = {
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
}


def init_history_store() -> None:
    """Create the local SQLite store used for saved prediction history."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                label TEXT NOT NULL,
                risk_pct REAL,
                support_level TEXT NOT NULL,
                school TEXT,
                sex TEXT,
                age INTEGER,
                failures_mat INTEGER,
                failures_por INTEGER,
                absences_mat INTEGER,
                absences_por INTEGER,
                studytime_mat INTEGER,
                studytime_por INTEGER,
                form_data TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_history_connection() -> sqlite3.Connection:
    """Return a SQLite connection configured for dict-like row access."""
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def parse_int_or_none(value: str | None) -> int | None:
    """Convert a submitted field value to int when possible."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_risk_value(prediction: int, risk_pct: float | None) -> float:
    """Normalize the displayed risk percentage whether probability exists or not."""
    if risk_pct is not None:
        return float(risk_pct)
    return 100.0 if prediction == 1 else 0.0


def derive_support_level(risk_value: float) -> str:
    """Group probability scores into practical support levels."""
    if risk_value >= 70:
        return "Urgent support"
    if risk_value >= 40:
        return "Targeted support"
    return "Routine monitoring"


def format_history_timestamp(value: str) -> str:
    """Format an ISO timestamp for history cards."""
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return value


def save_prediction_history(form_data: dict, prediction: int, label: str, risk_pct: float | None) -> None:
    """Persist a prediction record and its key input snapshot."""
    risk_value = compute_risk_value(prediction, risk_pct)
    payload = (
        datetime.now().isoformat(timespec="seconds"),
        prediction,
        label,
        risk_pct,
        derive_support_level(risk_value),
        form_data.get("school"),
        form_data.get("sex"),
        parse_int_or_none(form_data.get("age")),
        parse_int_or_none(form_data.get("failures_mat")),
        parse_int_or_none(form_data.get("failures_por")),
        parse_int_or_none(form_data.get("absences_mat")),
        parse_int_or_none(form_data.get("absences_por")),
        parse_int_or_none(form_data.get("studytime_mat")),
        parse_int_or_none(form_data.get("studytime_por")),
        json.dumps(form_data, sort_keys=True),
    )

    with get_history_connection() as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                created_at,
                prediction,
                label,
                risk_pct,
                support_level,
                school,
                sex,
                age,
                failures_mat,
                failures_por,
                absences_mat,
                absences_por,
                studytime_mat,
                studytime_por,
                form_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        conn.commit()


def load_prediction_history(limit: int = 100) -> list[dict]:
    """Load recent saved predictions for the history page."""
    with get_history_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM prediction_history
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    entries = []
    for row in rows:
        entry = dict(row)
        entry["form_data"] = json.loads(entry["form_data"])
        entry["created_at_display"] = format_history_timestamp(entry["created_at"])
        entry["risk_value"] = compute_risk_value(entry["prediction"], entry["risk_pct"])
        entry["risk_pct_display"] = f"{entry['risk_value']:.1f}%"
        entry["sex_display"] = {"F": "Female", "M": "Male"}.get(entry["sex"], entry["sex"] or "Unknown")
        entries.append(entry)

    return entries


def build_history_stats(entries: list[dict]) -> dict:
    """Summarize the saved prediction history for dashboard cards."""
    total_predictions = len(entries)
    at_risk_count = sum(1 for entry in entries if entry["prediction"] == 1)
    urgent_count = sum(1 for entry in entries if entry["risk_value"] >= 70)
    average_risk_pct = round(
        sum(entry["risk_value"] for entry in entries) / total_predictions, 1
    ) if total_predictions else None

    return {
        "total_predictions": total_predictions,
        "at_risk_count": at_risk_count,
        "not_at_risk_count": total_predictions - at_risk_count,
        "urgent_count": urgent_count,
        "average_risk_pct": average_risk_pct,
    }


def encode_value(field_name: str, value: str) -> int:
    """Encode a categorical field using the saved encoder when available."""
    if field_name in label_encoders:
        encoder = label_encoders[field_name]
        try:
            return int(encoder.transform([value])[0])
        except ValueError as exc:
            expected = list(getattr(encoder, "classes_", []))
            raise ValueError(
                f"Unexpected value '{value}' for field '{field_name}'. "
                f"Expected one of: {expected}"
            ) from exc

    field_base = field_name.rsplit("_", 1)[0] if field_name.endswith(("_mat", "_por")) else field_name

    if field_base in YESNO_FIELDS:
        mapping = ENCODE["yesno"]
    elif field_base in ENCODE:
        mapping = ENCODE[field_base]
    else:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Unknown field '{field_name}' with value '{value}'") from exc

    if value not in mapping:
        raise ValueError(
            f"Unexpected value '{value}' for field '{field_name}'. "
            f"Expected one of: {list(mapping.keys())}"
        )

    return mapping[value]


def build_feature_row(form: dict) -> pd.DataFrame:
    """
    Convert raw form data into a single-row DataFrame with the exact
    feature order expected by the trained model.
    """
    row = {}

    for field in [
        "school",
        "sex",
        "address",
        "famsize",
        "Pstatus",
        "Mjob",
        "Fjob",
        "reason",
        "nursery",
        "internet",
    ]:
        row[field] = encode_value(field, form[field])

    row["age"] = int(form["age"])
    row["Medu"] = int(form["Medu"])
    row["Fedu"] = int(form["Fedu"])

    for subject in ("mat", "por"):
        suffix = f"_{subject}"

        for field in [
            "traveltime",
            "studytime",
            "failures",
            "famrel",
            "freetime",
            "goout",
            "Dalc",
            "Walc",
            "health",
            "absences",
        ]:
            row[f"{field}{suffix}"] = int(form[f"{field}{suffix}"])

        for field in ["schoolsup", "famsup", "paid", "activities", "higher", "romantic"]:
            col = f"{field}{suffix}"
            row[col] = encode_value(col, form[col])

        guardian_col = f"guardian{suffix}"
        row[guardian_col] = encode_value(guardian_col, form[guardian_col])

    row["parent_edu_avg"] = (row["Medu"] + row["Fedu"]) / 2
    row["lifestyle_score"] = row["goout_mat"] + row["freetime_mat"]

    df = pd.DataFrame([row])
    return df[feature_order]


try:
    init_history_store()
    print(f"[OK] History store ready: {HISTORY_DB_PATH}")
except Exception as exc:
    history_store_error = str(exc)
    print(f"[ERROR] Could not initialize history store: {exc}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if load_error:
        return render_template("error.html", message=f"Model failed to load: {load_error}")

    if request.method == "GET":
        return render_template("predict.html")

    try:
        form_data = request.form.to_dict()
        df = build_feature_row(form_data)

        prediction = int(model.predict(df)[0])
        label = "At Risk" if prediction == 1 else "Not At Risk"
        risk_pct = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            classes = [int(cls) for cls in model.classes_]
            risk_idx = classes.index(1) if 1 in classes else 1
            risk_pct = round(float(proba[risk_idx]) * 100, 1)

        if not history_store_error:
            try:
                save_prediction_history(form_data, prediction, label, risk_pct)
            except Exception:
                app.logger.error("Failed to save prediction history\n%s", traceback.format_exc())

        return render_template(
            "result.html",
            label=label,
            prediction=prediction,
            risk_pct=risk_pct,
            form_data=form_data,
            history_enabled=history_store_error is None,
        )

    except ValueError as exc:
        return render_template("error.html", message=f"Input validation error: {exc}")
    except Exception as exc:
        app.logger.error(traceback.format_exc())
        return render_template("error.html", message=f"Prediction failed: {exc}")


@app.route("/insights")
def insights():
    return render_template("insights.html")


@app.route("/history")
def history():
    if history_store_error:
        return render_template("error.html", message=f"Prediction history is unavailable: {history_store_error}")

    try:
        entries = load_prediction_history(limit=100)
        stats = build_history_stats(entries)
        return render_template("history.html", entries=entries, stats=stats)
    except Exception as exc:
        app.logger.error(traceback.format_exc())
        return render_template("error.html", message=f"Could not load prediction history: {exc}")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
