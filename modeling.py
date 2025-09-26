# modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_and_evaluate(df, target_col, test_size=0.3, criterion="gini", max_depth=None, random_state=42):
    """
    df : dataframe sudah dipreproses (numeric)
    target_col : nama kolom label
    returns: (model, results_dict)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    labels = sorted(np.unique(y).tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(labels)>1 else None
    )

    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    results = {
        "model": model,
        "accuracy": acc,
        "report_str": report,
        "confusion_matrix": cm,
        "labels": labels,
        "feature_names": feature_names
    }
    return model, results

def predict_single(model, X_row):
    """X_row: 2D array-like (1, n_features)"""
    return model.predict(X_row)

def save_model(model, path="decision_tree_model.joblib"):
    joblib.dump(model, path)
    return path
