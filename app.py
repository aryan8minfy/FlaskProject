from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load your trained Random Forest model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route("/", methods=["GET"])
def home():
    return render_template("index 1.html", prediction=None)

# Predict from form submission
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve form fields
        Age = int(request.form["Age"])
        Experience = int(request.form["Experience"])
        Income = int(request.form["Income"])
        Family = int(request.form["Family"])
        CCAvg = float(request.form["CCAvg"])
        Education = int(request.form["Education"])
        Mortgage = int(request.form["Mortgage"])
        Securities_Account = int(request.form["Securities Account"])
        CD_Account = int(request.form["CD Account"])
        Online = int(request.form["Online"])
        CreditCard = int(request.form["CreditCard"])

        # Combine into NumPy array in correct order
        features = np.array([
            Age,
            Experience,
            Income,
            Family,
            CCAvg,
            Education,
            Mortgage,
            Securities_Account,
            CD_Account,
            Online,
            CreditCard
        ]).reshape(1, -1)

        # Prediction
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0].max()

        # Map prediction if needed
        pred_label = "Will Default" if pred == 1 else "No Default"

        return render_template(
            "index 1.html",
            prediction=pred_label,
            probability=f"{prob:.2f}"
        )

    except Exception as e:
        return f"Error during prediction: {e}"

# Batch predict from CSV
@app.route("/batch_predict", methods=["GET"])
def batch_predict():
    try:
        # Load new dataset
        df = pd.read_csv("New Customer Bank_Personal_Loan.csv")

        # Example: assume same columns as model was trained on
        # You may need to adjust this selection
        feature_cols = [
            "Age", "Experience", "Income", "Family",
            "CCAvg", "Education", "Mortgage",
            "Securities Account", "CD Account", "Online", "CreditCard"
        ]
        X_new = df[feature_cols]

        preds = model.predict(X_new)
        probs = model.predict_proba(X_new).max(axis=1)

        df["Prediction"] = np.where(preds == 1, "Will Default", "No Default")
        df["Probability"] = probs

        # Save results to new CSV
        output_path = "batch_predictions.csv"
        df.to_csv(output_path, index=False)

        return f"Batch predictions completed.<br>Results saved to <b>{output_path}</b>."
    except Exception as e:
        return f"Error during batch prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
