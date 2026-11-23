
from flask import Flask, render_template, request
import pandas as pd
import joblib
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

# Load trained machine learning model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "credit_model.pkl")

model = joblib.load(MODEL_PATH)

# Exact features expected by your trained model
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# Features represented by integer category codes
INTEGER_FEATURES = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

# Create database
os.makedirs("database", exist_ok=True)

def init_database():
    connection = sqlite3.connect("database/predictions.db")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_data TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL,
            created_at TEXT NOT NULL
        )
    """)
    connection.commit()
    connection.close()

init_database()


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        customer_data = {}

        for feature in FEATURES:
            value = request.form.get(feature)

            if value is None or value.strip() == "":
                return "Please complete all 23 input fields.", 400

            number = float(value)

            if not pd.notna(number):
                return "Invalid numerical input.", 400

            if feature in INTEGER_FEATURES:
                if not number.is_integer():
                    return f"{feature} must be a whole number.", 400
                customer_data[feature] = int(number)
            else:
                customer_data[feature] = number

        # Arrange inputs in the exact training order
        input_data = pd.DataFrame(
            [[customer_data[f] for f in FEATURES]],
            columns=FEATURES
        )

        # Generate prediction
        prediction = model.predict(input_data)[0]

        # Obtain probability of class 1 where supported
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            probability = float(probabilities[classes.index(1)])

        # Class 1 is assumed to mean default.
        # Verify this against your original target variable.
        if prediction == 1:
            result = "High Credit Risk (Default)"
        else:
            result = "Low Credit Risk (Non-Default)"

        # Save prediction
        connection = sqlite3.connect("database/predictions.db")
        connection.execute(
            """INSERT INTO predictions
            (customer_data, prediction, probability, created_at)
            VALUES (?, ?, ?, ?)""",
            (
                str(customer_data),
                result,
                probability,
                datetime.now().isoformat(timespec="seconds")
            )
        )
        connection.commit()
        connection.close()

        return render_template(
            "result.html",
            result=result,
            probability=probability,
            prediction=int(prediction)
        )

    except Exception as error:
        return f"Prediction error: {error}", 500


@app.route("/history")
def history():
    connection = sqlite3.connect("database/predictions.db")
    connection.row_factory = sqlite3.Row

    records = connection.execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    ).fetchall()

    connection.close()

    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
