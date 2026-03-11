# 🔍 Null-Hunter: Data Health & Preprocessing Dashboard

> **Live Demo → [null-hunter-dashboard.streamlit.app](https://null-hunter-dashboard.streamlit.app)**

---

## What Is Null-Hunter?

Null-Hunter is an interactive data quality dashboard built with Streamlit. Upload any CSV or Excel file and it will immediately audit your dataset for missing values, score its overall health, and walk you through multiple imputation strategies to clean it — all without writing a single line of code.

---

## Why I Built It

Most data science education focuses on choosing algorithms, tuning hyperparameters, evaluating accuracy. But in every real project, before any of that happens, you have to answer a more fundamental question: *is this data even usable?*

I built what I called Null-Hunter to close that gap between theory and practice. Understanding *why* data goes missing, and *which* strategy actually fits the problem, is one of the most consequential decisions in any ML pipeline. 

This project is my attempt to make that decision-making process visible, interactive, and grounded in something you can actually run on your own data.

---
## Features

### Data Health Audit
A missingness matrix (powered by `missingno`) and a per-column null count table that make the structure of missing data immediately visible — whether it's random, clustered, or systemic.

### Data Quality Score
A 0–100 composite score across four dimensions:

| Dimension | Weight | What It Measures |
|---|---|---|
| Completeness | 40% | Overall proportion of non-null cells, amplified so moderate missingness meaningfully hurts the score |
| Uniqueness | 20% | Penalises duplicate rows |
| Consistency | 20% | Flags columns where more than 5% of values are null |
| Validity | 20% | Flags columns where more than 10% of values are null |

The score updates live as you clean, so you can track improvement in real time.

### Imputation Strategies

**Simple Imputation (Mean / Median / Most Frequent)**
Best for sparse, randomly scattered nulls in columns that have no meaningful relationship to the rest of the dataset. Categorical columns are automatically restricted to Most Frequent.

**KNN Imputation**
Uses the K-Nearest Neighbors algorithm to fill each missing value based on the most similar rows. Best when missing values occur in columns that are correlated with other numeric columns — for example, weight and BMI missing together in a patient dataset.

**MICE (Multiple Imputation by Chained Equations)**
The most sophisticated option. Models each column with missing values as a regression function of all other columns, iterating until convergence. Best for datasets where multiple columns have interrelated missingness — for example, income, credit score, and debt-to-income ratio all missing together in a housing dataset.

**Drop Duplicates**
Removes exact duplicate rows before imputation to avoid inflating patterns in the data.

---

## Sample Datasets

Three datasets are included in `/sample_data` to demonstrate when each strategy is appropriate. Load them directly from the app to see the scoring system in action.

| File | Quality Score | Recommended Strategy | Characteristics |
|---|---|---|---|
| `survey_responses.csv` | 🟢 ~91% | Simple (Mean / Most Frequent) | Only 2 columns with sparse, uncorrelated nulls |
| `patient_vitals.csv` | 🟡 ~66% | KNN | 6 correlated numeric columns with 12–30% null rates |
| `housing_market.csv` | 🔴 ~44% | MICE | 10 columns with heavy interrelated nulls + 40 duplicate rows |

The key difference between the three: `survey_responses` has random missingness that no other column can predict, so the simplest strategy wins. `patient_vitals` has correlated features where KNN can find similar patients to borrow values from. `housing_market` has chained missingness where income missing often means credit score and DTI are also missing — only MICE handles that chain correctly.

---

## Getting Started

### Run it live
**[null-hunter-dashboard.streamlit.app](https://null-hunter-dashboard.streamlit.app)** — no installation required.

### Run it locally

```bash
git clone https://github.com/adityakm100/null-hunter-dashboard.git
cd null-hunter
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Requirements

```
streamlit
pandas
numpy
matplotlib
missingno
scikit-learn
openpyxl
```

---

## How to Use

1. **Upload your dataset** via the sidebar — CSV and Excel are both supported
2. **Read the audit** — check the null count table and missingness matrix to understand where data is missing and how much
3. **Check the quality score** — the breakdown chart shows which dimension is pulling your score down
4. **Pick a strategy** using the guide below:
   - Sparse nulls, no correlation to other columns → **Simple**
   - Nulls in correlated numeric columns → **KNN**
   - Nulls spread across multiple financially or biologically linked columns → **MICE**
5. **Apply imputation** and watch the quality score update
6. **Download** the cleaned dataset with the button at the bottom

---

## Project Structure

```
null-hunter/
├── app.py
├── requirements.txt
├── README.md
└── sample_data/
    ├── survey_responses.csv
    ├── patient_vitals.csv
    └── housing_market.csv
```

---

## Tech Stack

| Library | Role |
|---|---|
| Streamlit | Dashboard framework and session state management |
| pandas | Data loading, manipulation, and null detection |
| scikit-learn | KNNImputer and IterativeImputer (MICE) |
| missingno | Missingness matrix visualization |
| matplotlib | Quality score breakdown chart |

---

## Concepts Demonstrated

- Missingness mechanisms: MCAR, MAR, and MNAR
- Tradeoffs between single-pass and iterative imputation
- KNN imputation for correlated feature spaces
- MICE for chained, interrelated missingness
- Composite data quality scoring with weighted dimensions
- Streamlit session state for stateful, multi-step UI

## Connect
[LinkedIn](https://linkedin.com/in/aditya-kanginaya) · [GitHub](https://github.com/adityakm100)
