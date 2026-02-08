import json

import joblib
import pandas as pd
from flask import Flask, render_template, request

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = f"{ARTIFACTS_DIR}/model.joblib"
SCALER_PATH = f"{ARTIFACTS_DIR}/scaler.joblib"
FEATURES_PATH = f"{ARTIFACTS_DIR}/features.json"

app = Flask(__name__)

with open(FEATURES_PATH, "r", encoding="utf-8") as handle:
    FEATURES = json.load(handle)

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)


def build_recommendations(values: dict[str, float]) -> list[str]:
    recommendations: list[str] = []

    glucose = values.get("Glucose", 0.0)
    bmi = values.get("BMI", 0.0)
    bp = values.get("BloodPressure", 0.0)
    age = values.get("Age", 0.0)
    pedigree = values.get("DiabetesPedigreeFunction", 0.0)
    insulin = values.get("Insulin", 0.0)

    if glucose >= 140:
        recommendations.append(
            "Glucose is high; consider discussing an A1C or fasting glucose test with a clinician."
        )
    elif glucose >= 100:
        recommendations.append(
            "Glucose is above normal; consistent diet and activity habits can help, then recheck."
        )

    if bmi >= 30:
        recommendations.append(
            "BMI is in the obesity range; a structured nutrition and activity plan can help."
        )
    elif bmi >= 25:
        recommendations.append(
            "BMI is above the healthy range; small weight changes can improve glucose control."
        )

    if bp >= 90:
        recommendations.append(
            "Diastolic blood pressure looks elevated; regular monitoring is recommended."
        )

    if age >= 45:
        recommendations.append(
            "Risk increases with age; keep regular screenings with a healthcare professional."
        )

    if pedigree >= 0.8:
        recommendations.append(
            "Family history risk appears higher; share this with your clinician."
        )

    if insulin >= 200:
        recommendations.append(
            "Insulin is high; discuss results and medication options with a clinician."
        )

    recommendations.append(
        "This tool is for educational use only and is not a medical diagnosis."
    )

    return recommendations


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        values = {}
        errors = {}

        for feature in FEATURES:
            raw_value = request.form.get(feature, "").strip()
            if raw_value == "":
                errors[feature] = "Required"
                continue
            try:
                values[feature] = float(raw_value)
            except ValueError:
                errors[feature] = "Must be a number"

        if errors:
            return render_template("index.html", values=values, errors=errors)

        input_df = pd.DataFrame([values], columns=FEATURES)
        scaled = SCALER.transform(input_df)
        prediction = MODEL.predict(scaled)[0]

        is_diabetic = int(prediction) == 1
        verdict = "Diabetic" if is_diabetic else "Not diabetic"
        recommendations = build_recommendations(values) if is_diabetic else []
        return render_template(
            "result.html",
            verdict=verdict,
            values=values,
            recommendations=recommendations,
            is_diabetic=is_diabetic,
        )

    return render_template("index.html", values={}, errors={})


if __name__ == "__main__":
    app.run(debug=True)
