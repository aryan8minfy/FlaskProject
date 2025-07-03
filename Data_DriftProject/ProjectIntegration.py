import pandas as pd

import mlflow

from evidently.report import Report

from evidently.metric_preset import DataDriftPreset

from evidently.metrics import ColumnDriftMetric
 

# 📁 Load datasets

train_data = pd.read_csv("dataset loan.csv").drop(columns=["Personal Loan"])

test_data = pd.read_csv("New Customer Bank_Personal_Loan.csv")  # 15-record new dataset
 
# 📁 Set experiment

mlflow.set_experiment("Drift Metrics Analysis")
 
def log_data_drift_metrics(reference_df, current_df, run_name):

    from evidently.report import Report

    from evidently.metric_preset import DataDriftPreset

    import mlflow
 
    # Create and run the report

    drift_report = Report(metrics=[DataDriftPreset()])

    drift_report.run(reference_data=reference_df, current_data=current_df)
 
    # Save report as HTML

    drift_report.save_html("drift_report.html")
 
    # Get report as dict

    report_dict = drift_report.as_dict()
 
    # Start MLflow logging

    mlflow.set_experiment("Evidently_Drift_Metrics")

    with mlflow.start_run(run_name=run_name):

        # Log the HTML artifact

        mlflow.log_artifact("drift_report.html")
 
        # Loop through the metrics to find drift table

        for metric in report_dict["metrics"]:

            if metric.get("metric") == "DataDriftTable":

                result = metric["result"]
 
                # Log normalized drifted column count

                n_drifted = result.get("number_of_drifted_columns", 0)

                n_total = result.get("number_of_columns", 1)

                drift_ratio = n_drifted / n_total

                mlflow.log_metric("drifted_column_count", drift_ratio)
 
                # Log per-column drift score

                for feature, stats in result["drift_by_columns"].items():

                    score = stats.get("drift_score", 0)

                    mlflow.log_metric(f"drift_{feature}", score)

 
# Load and split

train_data = pd.read_csv("dataset loan.csv").drop(columns=["Personal Loan"])

train_labels = pd.read_csv("dataset loan.csv")["Personal Loan"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)
 
# Run 1: Train vs Test

log_data_drift_metrics(reference_df=X_train, current_df=X_test, run_name="Train_vs_Test_UnderstandingOf_DataDrift")
 
# Run 2: Historical vs New

historical_df = pd.read_csv("dataset loan.csv").drop(columns=["Personal Loan"])

new_df = pd.read_csv("New Customer Bank_Personal_Loan.csv")

log_data_drift_metrics(reference_df=historical_df, current_df=new_df, run_name="Historical_vs_New_DataDrift_Understanding")

 