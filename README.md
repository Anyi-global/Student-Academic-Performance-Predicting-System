# Student Academic Risk Prediction System

A Flask prototype that predicts whether a student is `At Risk` or `Not At Risk` academically so educators can intervene earlier.

---

## Purpose

The app combines demographic, family, social, and academic inputs from both Mathematics and Portuguese courses to produce a student risk classification and a risk probability.

---

## Folder Structure

```text
student_risk_app/
|-- app.py
|-- requirements.txt
|-- README.md
|-- model/
|   |-- student_risk_model.pkl
|   |-- label_encoders.pkl
|   |-- feature_order.pkl
|   |-- training_report.json
|   |-- backups/
|   `-- student-performance-data/
|       `-- dataset/
|           |-- student-mat.csv
|           `-- student-por.csv
|-- scripts/
|   |-- retrain_student_risk_model.py
|   `-- validate_model_contract.py
|-- templates/
|   |-- base.html
|   |-- index.html
|   |-- predict.html
|   |-- result.html
|   |-- insights.html
|   |-- about.html
|   `-- error.html
`-- static/
    |-- style.css
    `-- img/
```

---

## Setup

### 1. Open the project

```bash
cd student_risk_app
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Confirm the core model artifacts exist

Make sure these files are present inside `model/`:

| File | Description |
|------|-------------|
| `student_risk_model.pkl` | Trained decision tree classifier used by the app |
| `label_encoders.pkl` | Fitted label encoders for categorical input fields |
| `feature_order.pkl` | Exact 49-column feature order expected by the model |

Optional but useful:

| File | Description |
|------|-------------|
| `training_report.json` | Metrics and training settings from the latest retraining run |

---

## Running the App

```bash
python app.py
```

Then open:

`http://localhost:5000`

---

## Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page with overview and call to action |
| Predict | `/predict` | Student data entry form |
| Result | `/predict` (POST) | Risk label, probability, and recommendations |
| Insights | `/insights` | Explanatory charts and patterns |
| About | `/about` | System and model overview |

---

## Example Workflow

1. Open `http://localhost:5000`.
2. Go to **Predict**.
3. Fill in the common fields, Mathematics fields, and Portuguese fields.
4. Submit the form.
5. Review the result:
   - `At Risk` means the model found stronger academic risk signals.
   - `Not At Risk` means the student currently appears lower risk.
6. Use the probability and recommendations as decision-support, not as a final judgment.

---

## Engineered Features

These are computed automatically in `app.py` and are not entered manually:

| Feature | Formula |
|---------|---------|
| `parent_edu_avg` | `(Medu + Fedu) / 2` |
| `lifestyle_score` | `goout_mat + freetime_mat` |

---

## Current Model Details

- Algorithm: `DecisionTreeClassifier`
- Training data: UCI Student Performance dataset, with Math and Portuguese records merged
- Input features: 49
- Output labels: `1 = At Risk`, `0 = Not At Risk`
- Categorical encoding: 24 fitted label encoders stored in `label_encoders.pkl`
- Current training settings:
  - `class_weight="balanced"`
  - `random_state=42`
  - `max_depth=6`
  - `min_samples_leaf=5`
  - `ccp_alpha=0.0`

This model was retrained to produce softer, more realistic probabilities instead of only hard `0%` or `100%` outputs.

---

## Retraining and Validation

Two helper scripts are included for safe model updates.

### Retrain the model

```bash
python scripts/retrain_student_risk_model.py
```

What it does:

- loads the source CSV files from `model/student-performance-data/dataset/`
- rebuilds the merged training set
- fits label encoders for categorical columns
- trains the current decision tree configuration
- backs up the previous artifacts into `model/backups/<timestamp>/`
- writes updated artifacts into `model/`
- writes a summary report to `model/training_report.json`

### Validate the app-model contract

```bash
python scripts/validate_model_contract.py
```

What it checks:

- the model loads successfully
- the app still builds a valid `1 x 49` feature row
- prediction and `predict_proba()` still work
- the saved artifacts still match the Flask app contract

---

## Notes on Compatibility

- Keep the runtime `scikit-learn` version compatible with the version used to create the saved `.pkl` files.
- If you retrain the model, do not change the deployed contract unless you also update `app.py`:
  - feature names and order
  - categorical value mappings
  - target meaning (`0` and `1`)

---

## Important Note

This tool is a decision-support aid only. Predictions should be interpreted alongside teacher feedback, direct observation, and professional judgment.
