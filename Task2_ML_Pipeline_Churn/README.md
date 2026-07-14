# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

**Notebook:** [churn_pipeline.ipynb](churn_pipeline.ipynb) (full problem statement, preprocessing, GridSearchCV training, evaluation, confusion matrix/ROC visualizations with executed outputs)

## Objective
Build a reusable, production-ready ML pipeline to predict customer churn.

## Dataset
[Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
(7,043 customers, 20 features + churn label), saved locally at `data/Telco-Customer-Churn.csv`.

## Methodology / Approach
1. Load and clean the data: drop `customerID`, coerce `TotalCharges` to numeric (a few blanks for
   new customers become `NaN` and are median-imputed), encode `Churn` as 0/1.
2. Build a `ColumnTransformer` inside a scikit-learn `Pipeline`:
   - Numeric features: median imputation + standard scaling.
   - Categorical features: most-frequent imputation + one-hot encoding.
3. Train two classifiers end-to-end through the same pipeline: **Logistic Regression** and
   **Random Forest**.
4. Tune hyperparameters for each with `GridSearchCV` (5-fold CV, scored on F1).
5. Evaluate both models on a held-out test set (accuracy, F1, ROC-AUC) and keep the best.
6. Export the complete winning pipeline (preprocessing + model) with `joblib` for reuse
   without re-fitting.
7. Serve live predictions via a **Streamlit** form.

## How to Run

```bash
pip install -r requirements.txt

# Train, tune, evaluate, and export the pipeline
python train_pipeline.py

# Launch the interactive demo
streamlit run app.py
```

## Results
See `eval_results.txt` (generated after training) for per-model accuracy/F1/ROC-AUC and best
hyperparameters found by GridSearchCV.

## Key Results / Insights
- Wrapping preprocessing and modeling in a single `Pipeline` prevents data leakage during
  cross-validation and makes the exported artifact directly usable on raw, unprocessed input.
- Random Forest and Logistic Regression are compared under identical preprocessing to isolate the
  effect of model choice from feature engineering.
- The exported `churn_pipeline.joblib` is a single, self-contained object — no separate
  scaler/encoder files to track — demonstrating production-readiness.

## Skills Gained
- ML pipeline construction
- Hyperparameter tuning with GridSearchCV
- Model export and reusability (joblib)
- Production-readiness practices
