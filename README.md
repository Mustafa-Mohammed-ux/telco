# Customer Churn Prediction (Logistic Regression)

Predicts which telecom customers are likely to cancel their service, so a company could offer them a retention deal before they leave.

## Problem
Missing a customer who is about to churn costs more than sending an unnecessary retention offer. The model is therefore judged mainly on **recall for the churn class**, balanced against precision, rather than on accuracy.

## Dataset
Telco Customer Churn (IBM sample data, Kaggle): 7,043 customers, 20 features covering demographics, subscribed services, contract and billing, with about 27% churners.

## Approach
- Cleaned the data and encoded features (binary mapping, ordinal `Contract`, one-hot for internet service and payment method).
- Found and fixed perfectly redundant dummy columns caused by "No internet service" appearing in six columns.
- Trained logistic regression with and without feature scaling (scaling made almost no difference).
- Handled class imbalance with threshold tuning and `class_weight='balanced'`.

## Results (held-out test set, 1,409 customers, 373 churners)
| Model | Precision | Recall | Missed churners (FN) | False alarms (FP) |
|---|---|---|---|---|
| Baseline (threshold 0.5) | 68.5% | 59.5% | 151 | 102 |
| **Final: balanced class weights, threshold 0.5** | **51.3%** | **82.3%** | **66** | **291** |

The final model catches 307 of 373 churners, cutting missed churners by 56% versus the baseline. About half of the customers it flags really do churn, versus a 26.5% base rate, so targeting is roughly 1.9x more efficient than contacting customers at random. Accuracy drops from 82% to 75% by design, because the baseline looked good on accuracy only by ignoring most churners.

## Key findings
- Contract type and tenure are the strongest predictors of churn; gender has no meaningful effect.
- Fiber optic internet and electronic-check payment are linked to higher churn.
- Class weighting and threshold tuning both trade precision for recall along a similar curve.

## Limitations
- Thresholds were compared on the same test set used for the final numbers, so results are slightly optimistic.
- Single train/test split, no cross-validation.
- Recall was prioritized by reasoning, not from real retention-cost figures.

## Next steps
Stratified cross-validation, a cost-based threshold, non-linear models, and a Streamlit app that returns a churn probability for one customer.

## How to run
Place `telco_cust.csv` in the same folder and run `telco_customer_churn.ipynb`.
Requires: numpy, pandas, scikit-learn, matplotlib, seaborn.
