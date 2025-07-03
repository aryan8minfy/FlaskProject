# 📊 Data Drift Analysis Pipeline

This project demonstrates **data drift detection and monitoring** using **Evidently** and **MLflow**.

---

## 📝 Overview

The script performs the following:

1. Loads historical training data and new incoming data.
2. Generates data drift reports with Evidently.
3. Logs drift metrics and HTML reports into MLflow for experiment tracking.
4. Runs two drift comparisons:
   - **Train vs Test Split:** Understand if random splits have drift.
   - **Historical vs New Data:** Detect drift in production data.

---

## 🧩 Key Libraries

- **pandas** — Data loading & manipulation.
- **Evidently** — Data drift detection.
- **MLflow** — Experiment tracking & logging.
- **scikit-learn** — Data splitting.

---

## Screenshots
# Historical_vs_New_DataDrift_Understanding
<img width="542" alt="image" src="https://github.com/user-attachments/assets/46e3be21-ce63-4768-b802-a821f5055b87" />
<img width="905" alt="image" src="https://github.com/user-attachments/assets/230e1d70-12ab-4589-9dc6-0a0d2bf8f8ed" />

# Train_vs_Test_UnderstandingOf_DataDrift
<img width="948" alt="image" src="https://github.com/user-attachments/assets/292054d6-208e-4298-a683-44821eb6b445" />
<img width="939" alt="image" src="https://github.com/user-attachments/assets/b55c289f-8812-447f-994e-65f9ed5bc5ab" />




## 🚀 Dependencies
- pip install pandas mlflow evidently scikit-learn
- python ProjectIntegration.py
- mlflow ui

## 📈 Example Use Cases
✅ Monitor incoming data quality in production.
✅ Detect shifts in customer behavior over time.
✅ Automate drift detection in your ML pipeline.
