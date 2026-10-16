# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to cancel service (churn), so the company can contact at-risk customers with retention offers **before** they leave. Includes a Streamlit app that returns a churn probability for a new customer.

**Live demo:** https://telco-customer-churn-prediction-ml-project-app.streamlit.app/

## Business Problem

Every time a customer leaves, the company loses monthly revenue and the customer's future lifetime value. Acquiring a new customer costs more than retaining an existing one. This project builds a binary classifier (Churn = Yes/No) and an estimated churn probability to help the retention team prioritise outreach.

## Dataset

- **Source:** [Telco Customer Churn (Kaggle, BlastChar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn` (1 = left, 0 = stayed). About 26.5% of customers churned, so the data is imbalanced.
- **Features:** demographics, account info (tenure, contract, payment method), subscribed services, and monthly/total charges.

## Project Structure

```
churn-project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── model/
│   └── churn_pipeline.joblib      # fitted preprocessing + model
├── notebook/
│   └── notebook.ipynb             # full training workflow
├── app.py                         # Streamlit application
├── requirements.txt
├── README.md
└── .gitignore
```

## Methodology

1. **Cleaning:** `TotalCharges` was text because of 11 blank values. All 11 belonged to customers with `tenure = 0` (new customers with no bill yet), so they were filled with 0. No duplicate rows were found. `customerID` was excluded from the model features.
2. **EDA:** churn rate, distributions and outliers, and churn compared against contract, tenure, monthly charges, payment method, internet service and tech support.
3. **Split:** 80/20 train/test, `random_state=42`, `stratify=y`.
4. **Preprocessing:** a single scikit-learn `Pipeline` with `ColumnTransformer`.
   - Numeric: median imputation + `StandardScaler` (scaling used for Logistic Regression only)
   - Categorical: most-frequent imputation + `OneHotEncoder(handle_unknown='ignore')`
   - Everything is fitted on training data only, which avoids data leakage.
5. **Models compared:** Dummy baseline, Logistic Regression, Decision Tree, Random Forest.
6. **Tuning:** 5-fold stratified cross-validation and `GridSearchCV` (scoring = F1) over `C` and `class_weight`.
7. **Final evaluation:** the test set was used once for the final model.

## Key Findings (EDA)

- About 1 in 4 customers churned.
- Month-to-month contracts have a far higher churn rate (about 43%) than one-year (about 11%) or two-year (about 3%) contracts.
- Churners have a much lower median tenure, so most churn happens early in the relationship.
- Churners pay higher monthly charges on average.
- Electronic check users churn most (about 45%) compared with automatic payment users (about 15-17%).
- Fiber optic customers and customers without Tech Support churn more.

These are associations in the data, not proven causes.

## Model Comparison (test set, default settings)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 |
| Decision Tree | 0.717 | 0.468 | 0.481 | 0.474 | 0.641 |
| Random Forest | 0.782 | 0.613 | 0.487 | 0.542 | 0.819 |

The Dummy model reaches 73.5% accuracy while catching no churners, which shows that accuracy alone is misleading on this dataset.

## Final Model

**Logistic Regression** with `C=0.1` and `class_weight='balanced'`, saved as a complete pipeline in `model/churn_pipeline.joblib`.

| Metric | Test result |
|---|---|
| Accuracy | 0.742 |
| Precision | 0.509 |
| Recall | 0.786 |
| F1 | 0.618 |
| ROC-AUC | 0.841 |

Confusion matrix: TN = 751, FP = 284, FN = 80, TP = 294.

- Cross-validated F1 was 0.629, close to the test F1, which suggests stable performance.
- `class_weight='balanced'` raised recall from 0.559 to 0.783 and cut missed churners from 165 to 81, at the cost of lower precision and more false alarms. ROC-AUC did not change, so the ranking ability is the same and only the decision threshold moved.

## Business Recommendations

1. Offer incentives to move month-to-month customers to longer contracts.
2. Strengthen onboarding and follow-up during the first 3-6 months.
3. Encourage automatic payments (for example, a small cashback).
4. Bundle or trial Tech Support, especially for fiber optic customers.
5. Review pricing and fiber service quality.
6. Score customers regularly and contact the highest-risk group first.

## How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter the customer's details and click **Predict** to see the churn probability and risk label.

> **Important:** install the same `scikit-learn` version that was used to train the model (see `requirements.txt`). A different version can cause loading warnings or errors.

To retrain, open `notebook/notebook.ipynb` and run all cells. The CSV path inside the notebook is `../data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Limitations

- Based on a single dataset from one company, so results may not transfer to other markets.
- Precision is about 51%, so roughly half of the flagged customers are false alarms. The model suits cases where a retention offer costs less than losing a customer.
- Predicted probabilities are estimates, not guarantees.
- The app estimates `TotalCharges` as `tenure x MonthlyCharges`, which can differ from the real value.
- The data is a snapshot and has no information on complaints, network quality or competitor offers.
- Findings are associations, not causal proof.

## Future Work

- Try Gradient Boosting and other models.
- Tune the decision threshold using the real cost of a lost customer versus a retention offer.
- Add model explanations (feature importance or SHAP) to the app.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit