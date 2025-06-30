import streamlit as st
import pandas as pd
import pickle
import os

# 📁 Get folder path where this Python file is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📦 Load model, scaler, and column names
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))

# 🎨 Page configuration
st.set_page_config(
    page_title="💼 Loan Defaulter Prediction",
    page_icon="💼",
    layout="wide"
)

# 🏷️ Title
st.markdown(
    """
    <h1 style="text-align: center; color: #0e4d92;">
        💼 Loan Defaulter Prediction
    </h1>
    <p style="text-align: center; font-size:18px;">
        Upload customer data to predict potential loan defaulters.
    </p>
    """,
    unsafe_allow_html=True
)

# 📁 File uploader
uploaded_file = st.file_uploader("📂 **Upload CSV File**", type="csv")

if uploaded_file is not None:
    # 🟢 Read CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(data.head(), use_container_width=True)

    # ✨ Preprocessing
    data.columns = data.columns.str.strip()
    original_data = data.copy()

    # Drop unnecessary columns if present
    data.drop(columns=['ID', 'ZIP Code'], inplace=True, errors='ignore')

    # Remove rows with invalid Experience
    if 'Experience' in data.columns:
        data = data[data['Experience'] >= 0]

    # Create dummies for categorical variables
    data = pd.get_dummies(data, drop_first=True)

    # Ensure all expected columns exist
    for col in columns:
        if col not in data.columns:
            data[col] = 0

    # Reorder columns to match model
    data = data[columns]

    # 🔄 Scale and predict
    data_scaled = scaler.transform(data)
    preds = model.predict(data_scaled)

    # ✅ Add predictions to original data
    original_data["Loan Status"] = [
        "💥 Defaulter" if p == 1 else "✅ Non-Defaulter" for p in preds
    ]

    st.subheader("📈 Prediction Results")
    st.dataframe(original_data, use_container_width=True)

    # 📥 Download button
    csv_download = original_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_download,
        file_name="loan_predictions.csv",
        mime="text/csv"
    )

else:
    st.info(
        "👈 Upload a CSV file to get started.",
        icon="ℹ️"
    )
