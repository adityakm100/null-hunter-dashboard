import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import KNNImputer

st.set_page_config(page_title="Null-Hunter Dashboard", layout="wide")
st.title("🔍 Null-Hunter: Data Health & Preprocessing Check")

st.markdown("Automated data cleaning and missingness visualization tool for better quality AI/ML pipelines.")

#data quality helper
def compute_quality_score(df: pd.DataFrame) -> tuple[float, dict]:
    total_cells = df.shape[0] * df.shape[1]
    
    missing_ratio = df.isnull().sum().sum() / max(total_cells, 1)
    completeness = max(0.0, 1.0 - missing_ratio) * 100

    dup_ratio = df.duplicated().sum() / max(len(df), 1)
    uniqueness = max(0.0, 1.0 - dup_ratio) * 100

    clean_cols = sum(
        1 for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    )
    consistency = clean_cols / max(len(df.columns), 1) * 100

    bad_cols = sum(
        1 for col in df.columns
        if df[col].isnull().mean() > 0.5
    )
    validity = max(0.0, 1.0 - bad_cols / max(df.shape[1], 1)) * 20

    total = (completeness + uniqueness + consistency + validity) / 4
    breakdown = {
        "Completeness": round(completeness, 2),
        "Uniqueness": round(uniqueness, 2),
        "Consistency": round(consistency, 2),
        "Validity": round(validity, 2)
    }
    return round(total, 1), breakdown

def score_emoji(score: float) -> str:
    if score >= 80:
        return "🟢"
    elif score >= 55:
        return "🟡"
    return "🔴"


# Sidebar
st.sidebar.header("📥 Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if "df_working" not in st.session_state:
    st.session_state.df_working = None

if uploaded_file is not None:
    try:
        if st.session_state.get("loaded_file") != uploaded_file.name:
            df_original = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            if df_original.empty:
                st.error("The uploaded file is empty. Please upload a valid dataset.")
                st.stop()
            st.session_state.df_working = df_original.copy()
            st.session_state.loaded_file = uploaded_file.name
            st.success("Dataset loaded successfully!")
        else:
            st.success("Dataset loaded successfully!")

        df_to_modify = st.session_state.df_working

        # Audit
        st.subheader("📊 Data Health Audit")

        quality_score, breakdown = compute_quality_score(df_to_modify)
        emoji = score_emoji(quality_score)
        grade = "Excellent" if quality_score >= 80 else ("Needs Attention" if quality_score >= 55 else "Action Required")

        score_col, breakdown_col = st.columns([1, 2])
        with score_col:
            st.metric(label = f"{emoji} Overall Quality Score", value=f"{quality_score}%", delta=grade)
            st.caption(f"**Grade** {grade}")

        with breakdown_col:
            breakdown_df = pd.DataFrame(list(breakdown.items()), columns=["Dimension", "Score (%)"])
            fig, ax = plt.subplots(figsize=(6, 2.2))
            bars = ax.barh(breakdown_df["Dimension"], breakdown_df["Score (%)"], color=['#4CAF50', '#FFC107', '#2196F3', '#FF5722'])
            ax.set_xlim(0, 100)
            ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)
            ax.tick_params(left=False, bottom=False)
            ax.set_title("Score Breakdown", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)


        col1, col2 = st.columns([1, 2])

        with col1:
            st.write(f"**Dimensions:** {df_to_modify.shape[0]} rows x {df_to_modify.shape[1]} columns")
            nulls = df_to_modify.isnull().sum()
            st.dataframe(
                pd.DataFrame({"Column": nulls.index, "Null Count": nulls.values})
                .sort_values(by="Null Count", ascending=False)
            )

        with col2:
            fig, ax = plt.subplots(figsize=(10, 4))
            msno.matrix(df_to_modify, ax=ax, fontsize=12, sparkline=False)
            st.pyplot(fig)

        st.divider()

        # Preprocessing and imputation selection
        st.subheader("🛠️ Preprocessing & Imputation")

        cols_with_nulls = df_to_modify.columns[df_to_modify.isnull().any()].tolist()
        num_cols_with_nulls = df_to_modify[cols_with_nulls].select_dtypes(include=np.number).columns.tolist()
        cat_cols_with_nulls = df_to_modify[cols_with_nulls].select_dtypes(exclude=np.number).columns.tolist()

        imputation_method = st.selectbox("Choose Strategy for Missing Data Imputation:", [
            "None (Keep Nulls)",
            "Drop Duplicates",
            "Simple Imputation (Mean/Median/Most Frequent)",
            "Advanced KNN (K Nearest Neighbors) Imputation",
        ])

        if imputation_method == "Drop Duplicates":
            dupe_count = df_to_modify.duplicated().sum()
            if st.button(f"Clear {dupe_count} Duplicates"):
                # FIX 4: Write result back to session state
                st.session_state.df_working = df_to_modify.drop_duplicates().reset_index(drop=True)
                st.success("Duplicates removed successfully!")
                st.rerun()

        elif imputation_method == "Simple Imputation (Mean/Median/Most Frequent)":
            if not cols_with_nulls:
                st.info("No columns with missing values found.")
            else:
                col = st.selectbox("Select Column for Simple Imputation:", cols_with_nulls)
                if col in num_cols_with_nulls:
                    strat = st.selectbox("Select Imputation Strategy:", ["Mean", "Median", "Most Frequent"])
                else:
                    strat = st.selectbox("Select Imputation Strategy:", ["Most Frequent"])
                    st.info(f"**{col}** is a text/categorical column — only 'Most Frequent' strategy applies.")

                if st.button("Apply Simple Imputation"):
                    if strat == "Mean":
                        fill_value = df_to_modify[col].mean()
                    elif strat == "Median":
                        fill_value = df_to_modify[col].median()
                    else:
                        fill_value = df_to_modify[col].mode()[0]

                    st.session_state.df_working[col] = df_to_modify[col].fillna(fill_value)
                    st.success(f"Successfully filled nulls in **{col}** with **{strat}** → `{fill_value}`")
                    st.rerun()

        elif imputation_method == "Advanced KNN (K Nearest Neighbors) Imputation":
            st.info("KNN imputation considers the average similarity between rows to impute missing values. "
                    "The goal is to capture complex relationships in the data for more accurate imputation.")

            if not num_cols_with_nulls:
                st.info("No numeric columns with missing values found for KNN imputation.")
            else:
                st.write(f"Will impute these numeric columns: **{', '.join(num_cols_with_nulls)}**")
                if st.button("Apply KNN Imputation"):
                    imputer = KNNImputer(n_neighbors=5)
                    df_knn = st.session_state.df_working.copy()
                    df_knn[num_cols_with_nulls] = imputer.fit_transform(df_knn[num_cols_with_nulls])
                    st.session_state.df_working = df_knn
                    st.success(f"KNN imputation applied successfully to: {', '.join(num_cols_with_nulls)}!")
                    st.rerun()

        st.divider()

        st.download_button(
            "📥 Download Cleaned Dataset",
            data=st.session_state.df_working.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing the file: {e}")
        st.exception(e)

else:
    st.session_state.df_working = None
    st.session_state.pop("loaded_file", None)
    st.info("Awaiting for dataset upload. Check the sidebar!")