import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import KNNImputer, SimpleImputer

st.set_page_config(page_title="Null-Hunter Dashboard", layout="wide")
st.title("🔍 Null-Hunter: Data Health & Preprocessing Check")

st.markdown("Automated data cleaning and missingness visualization tool for better quality AI/ML pipelines.")

#sidebar
st.sidebar.header("📥 Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        if df.empty:
            st.error("The uploaded file is empty. Please upload a valid dataset.")
            st.stop()
        st.success("Dataset loaded successfully!")
        
        #audit
        st.subheader("📊 Data Health Audit")
        col1, col2 = st.columns([1,2])

        with col1:
            st.write(f"**Dimensions:** {df.shape[0]} rows x {df.shape[1]} columns")
            nulls = df.isnull().sum()
            st.dataframe(nulls[nulls > 0].sort_values(ascending=False).to_frame(name="Null Count").rename("Null Count"))
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 4))
            msno.matrix(df, ax = ax, fontsize = 12, sparkline = False)
            st.pyplot(fig)

        st.divider()

        #preprocessing and imputation selection
        st.subheader("🛠️ Preprocessing & Imputation")
        df_to_modify = df.copy()

        imputation_method = st.selectbox("Choose Strategy for Missing Data Imputation:", [
           "None (Keep Nulls)",
           "Drop Duplicates",
           "Simple Imputation (Mean/Median)",
           "Advanced KNN (K Nearest Neighbors) Imputation",
        ])

        if imputation_method == "Drop Duplicates":
            if st.button("Clear Duplicates"):
                df_to_modify = df_to_modify.drop_duplicates()
                st.success("Duplicates removed successfully!")
        
        elif imputation_method == "Simple Imputation (Mean/Median)":
            num_cols = df_to_modify.select_dtypes(include=np.number).columns
            col = st.selectbox("Select Column for Simple Imputation:", num_cols)
            strat = st.selectbox("Select Imputation Strategy:", ["Mean", "Median"])
            if st.button("Apply Simple Imputation"):
                imputer = SimpleImputer(strategy=strat.lower())
                df_to_modify[col] = imputer.fit_transform(df_to_modify[[col]])
                st.success(f"Simple imputation applied to '{col}' using {strat} strategy!")




