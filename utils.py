"""
Utility functions for the Rainfall Prediction Streamlit app.
Extracted from the Jupyter notebook pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from category_encoders import MEstimateEncoder
import sklearn.metrics as metrics


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def clean(df):
    """Rename misspelled column."""
    df = df.rename(columns={"temparature": "temperature"})
    return df


def impute(df):
    """Fill missing numeric values with mean, categorical with 'None'."""
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(df[name].mean())
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df


def load_data():
    """Load and clean train/test CSVs."""
    train_df = pd.read_csv("train.csv", index_col="id", header=0)
    test_df = pd.read_csv("test.csv", index_col="id", header=0)
    train_df = clean(train_df)
    test_df = clean(test_df)
    test_df = impute(test_df)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Feature-utility helpers
# ---------------------------------------------------------------------------

def get_mutual_info_scores(X, y, discrete_features):
    """Compute mutual-information scores against the target."""
    scores = mutual_info_classif(
        X, y,
        discrete_features=X.columns.isin(discrete_features.columns),
    )
    return scores


def drop_uninformative(df, mi_scores):
    """Drop columns whose MI score is 0."""
    return df.loc[:, mi_scores > 0.0]


# ---------------------------------------------------------------------------
# Outlier helpers
# ---------------------------------------------------------------------------

def mark_outliers(X):
    """Flag rows where sunshine > 10 AND cloud < 60."""
    df = pd.DataFrame(index=X.index)
    outliers = X[(X.sunshine > 10) & (X.cloud < 60)]
    df["outlier"] = X.index.isin(outliers.index).astype(int)
    return df


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_labels(df, n_clusters=5):
    """KMeans clustering: returns labels, centroid distances, inertia."""
    X = df.copy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    x_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    X_new = pd.DataFrame(index=df.index)
    X_new["cluster"] = kmeans.fit_predict(x_scaled)
    X_new["cluster"] = X_new["cluster"].astype("category")
    X_cd = kmeans.transform(x_scaled)
    X_cd = pd.DataFrame(
        X_cd,
        columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])],
        index=df.index,
    )
    return X_new, X_cd, kmeans.inertia_


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------

class CrossFoldEncoder:
    """K-fold target encoder wrapper."""

    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs
        self.cv_ = KFold(n_splits=5)

    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(X.iloc[idx_encode, :], y.iloc[idx_encode])
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    def transform(self, X):
        from functools import reduce
        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded


def day_to_month(df):
    """Convert day-of-year to month number."""
    dt = pd.DataFrame(index=df.index)
    dt["month"] = pd.to_datetime(df["day"], format="%j").dt.month
    return dt


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def interactions(X):
    df = pd.DataFrame(index=X.index)
    df["dewpoint_temperature"] = (X.dewpoint + X.temperature) * X.humidity
    return df


# ---------------------------------------------------------------------------
# Main feature-engineering pipeline
# ---------------------------------------------------------------------------

def create_features(df, df_test=None):
    """
    Full feature-engineering pipeline.
    If df_test is provided, returns (X_train, X_test); otherwise returns X.
    """
    X = df.copy()
    y = X.pop("rainfall")
    discrete_features = X.select_dtypes(include=["int64"])
    mi_scores = get_mutual_info_scores(X, y, discrete_features)

    if df_test is not None:
        X_test = df_test.copy()
        X = pd.concat([X, X_test])

    X = drop_uninformative(X, mi_scores)
    X = X.drop(columns=["mintemp", "maxtemp", "winddirection"], errors="ignore")

    X = X.join(interactions(X))
    X = X.join(mark_outliers(X))

    clusters_labels_df, cluster_distances, _ = cluster_labels(X, n_clusters=5)
    X = X.join(cluster_distances)

    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X = X.join(encoder.fit_transform(X, y, cols=["day"]))
    X.drop(columns=["day", "month"], inplace=True, errors="ignore")

    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))
        X_test.drop(columns=["day", "month"], inplace=True, errors="ignore")

    if df_test is not None:
        return X, X_test, encoder
    else:
        return X, None, encoder


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def make_results(model_name, model_object, metric):
    """Extract CV results from a GridSearchCV object."""
    metric_dict = {
        "auc": "mean_test_roc_auc",
        "precision": "mean_test_precision",
        "recall": "mean_test_recall",
        "f1": "mean_test_f1",
        "accuracy": "mean_test_accuracy",
    }
    cv_results = pd.DataFrame(model_object.cv_results_)
    best = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]
    table = pd.DataFrame(
        {
            "model": [model_name],
            "precision": [best.mean_test_precision],
            "recall": [best.mean_test_recall],
            "F1": [best.mean_test_f1],
            "accuracy": [best.mean_test_accuracy],
            "AUC": [best.mean_test_roc_auc],
        }
    )
    return table


def get_scores(model_name, model, X_test_data, y_test_data):
    """Compute validation scores from a fitted GridSearchCV."""
    preds = model.best_estimator_.predict(X_test_data)
    preds_proba = model.best_estimator_.predict_proba(X_test_data)[:, 1]
    table = pd.DataFrame(
        {
            "model": [model_name],
            "precision": [metrics.precision_score(y_test_data, preds)],
            "recall": [metrics.recall_score(y_test_data, preds)],
            "F1": [metrics.f1_score(y_test_data, preds)],
            "accuracy": [metrics.accuracy_score(y_test_data, preds)],
            "AUC": [metrics.roc_auc_score(y_test_data, preds_proba)],
        }
    )
    return table
