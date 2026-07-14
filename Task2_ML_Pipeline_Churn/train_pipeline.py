"""
Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
Builds a reusable, production-ready pipeline for predicting customer churn
on the Telco Customer Churn dataset.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/Telco-Customer-Churn.csv"
TARGET = "Churn"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=["customerID"], inplace=True)
    # TotalCharges has some blank strings for new customers; coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])


def main():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            },
        ),
    }

    best_overall = None
    best_overall_score = -1
    results = {}

    for name, (estimator, param_grid) in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])

        print(f"\nRunning GridSearchCV for {name}...")
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)

        preds = grid.predict(X_test)
        proba = grid.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        print(f"{name}: best_params={grid.best_params_}")
        print(f"{name}: accuracy={acc:.4f} f1={f1:.4f} roc_auc={auc:.4f}")
        print(classification_report(y_test, preds))

        results[name] = {"accuracy": acc, "f1": f1, "roc_auc": auc, "best_params": grid.best_params_}

        if f1 > best_overall_score:
            best_overall_score = f1
            best_overall = grid.best_estimator_
            best_overall_name = name

    print(f"\nBest model: {best_overall_name} (f1={best_overall_score:.4f})")
    joblib.dump(best_overall, "churn_pipeline.joblib")
    print("Saved pipeline to churn_pipeline.joblib")

    with open("eval_results.txt", "w") as f:
        f.write(str(results))
        f.write(f"\n\nBest model: {best_overall_name}\n")


if __name__ == "__main__":
    main()
