# Diabetes Prediction Web App

This project trains a diabetes prediction model and serves it through a simple Flask web application.

## Project Structure

- DiabetesPrediction.ipynb: Notebook for exploration, training, and evaluation.
- train_model.py: Trains the model and saves artifacts.
- app.py: Flask web app that loads artifacts and serves predictions.
- templates/index.html: Input form UI.
- templates/result.html: Result page with guidance.
- requirements.txt: Python dependencies.
- artifacts/: Saved model, scaler, and feature list.
- diabetes.csv: Dataset used for training.

## Setup

Create and activate a virtual environment (optional but recommended), then install dependencies:

```
pip install -r requirements.txt
```

## Train the Model

```
python train_model.py
```

This creates:
- artifacts/model.joblib
- artifacts/scaler.joblib
- artifacts/features.json

## Run the Web App

```
python app.py
```

Open the app in your browser:

```
http://127.0.0.1:5000
```

## Notes

- The web app expects artifacts to exist. If they are missing, rerun train_model.py.
- This tool is for educational use only and does not provide medical advice.
