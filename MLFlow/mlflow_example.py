import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# -------------------------------------------
# ✅ Step 1: Load & Prepare Dataset
# -------------------------------------------
data_path = "dataset loan (1).csv"
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip()

# Drop unwanted columns
df.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')

# Remove negative Experience
df = df[df['Experience'] >= 0]

# Define features & target
X = df.drop('Personal Loan', axis=1)
y = df['Personal Loan']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Set Experiment
mlflow.set_experiment("final_Bank_Personal_Loan_Prediction")

# -------------------------------------------
# ✅ Step 2: Function to Train, Log, and Register
# -------------------------------------------
def log_model_with_mlflow(name, model, param_grid, X_train, y_train, X_test, y_test):

    with mlflow.start_run(run_name=name):

        # Log dataset snapshot
        df.to_csv("input_data.csv", index=False)
        mlflow.log_artifact("input_data.csv")

        # Grid Search
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Log best parameters
        mlflow.log_params(grid.best_params_)

        # Cross-validation metrics
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
        mlflow.log_metric("cv_f1_mean", np.mean(cv_scores))
        mlflow.log_metric("cv_f1_std", np.std(cv_scores))

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_score", f1)

        print(f"\n📋 {name} Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        os.makedirs("plots", exist_ok=True)
        cm_path = f"plots/{name}_conf_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Log trained model artifact
        mlflow.sklearn.log_model(best_model, f"{name}_model")

        # ⭐️ REGISTER the model in the Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{name}_model"
        model_name = f"BankLoan_{name}_Model"

        # This registers the model (will create a new version or reuse existing)
        mlflow.register_model(model_uri, model_name)

        return best_model

# -------------------------------------------
# ✅ Step 3: Train & Log All Models
# -------------------------------------------

# Logistic Regression
log_params = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']}
log_model_with_mlflow(
    "LogisticRegression",
    LogisticRegression(max_iter=1000),
    log_params,
    X_train_res,
    y_train_res,
    X_test,
    y_test
)

# Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
log_model_with_mlflow(
    "RandomForest",
    RandomForestClassifier(random_state=42),
    rf_params,
    X_train_res,
    y_train_res,
    X_test,
    y_test
)

# XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5]
}
log_model_with_mlflow(
    "XGBoost",
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    xgb_params,
    X_train_res,
    y_train_res,
    X_test,
    y_test
)
import mlflow.sklearn
import pandas as pd
import pickle
 
# Step 1: Load registered model from MLflow
model = mlflow.sklearn.load_model("models:/BankLoan_XGBoost_Model/1")  # Adjust version as needed
 
# Step 2: Load scaler and columns
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
 
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)
 
# Step 3: Create sample input (ensure it matches training features)
sample_input = pd.DataFrame([[
    45,     # Age
    80000,  # Income
    2,      # Education
    100,    # Mortgage
    1.2,    # CCAvg
    4,      # Family
    1,      # CD Account
    1,      # CreditCard
    0,      # Online
    0,      # Securities Account
    20      # Experience
]], columns=columns)
 
# Step 4: Scale the input
scaled_input = scaler.transform(sample_input)
 
# Step 5: Predict
prediction = model.predict(scaled_input)
predicted_proba = model.predict_proba(scaled_input)
 
# Output
print("Prediction:", prediction[0])  # 0 = not default, 1 = default
print(f"Predicted Probabilities: {predicted_proba[0]}")