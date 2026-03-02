"""
Rainfall Prediction — Streamlit App
====================================
Converted from the Jupyter notebook pipeline.
Covers EDA, feature engineering, model training (Logistic Regression,
Decision Tree, Random Forest, XGBoost), model comparison, and
single-sample prediction.
"""

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from utils import (
    load_data,
    clean,
    impute,
    get_mutual_info_scores,
    drop_uninformative,
    mark_outliers,
    cluster_labels,
    CrossFoldEncoder,
    interactions,
    create_features,
    make_results,
    get_scores,
    day_to_month,
)

random.seed(0)
np.random.seed(0)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rainfall Prediction", layout="wide")
st.title("🌧️ Rainfall Prediction Pipeline")

# ── Cached data loading ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data …")
def cached_load_data():
    return load_data()


@st.cache_data(show_spinner="Engineering features …")
def cached_create_features(_train_df, _test_df=None):
    if _test_df is not None:
        X_train, X_test, encoder = create_features(_train_df, _test_df)
        return X_train, X_test, encoder
    else:
        X_train, _, encoder = create_features(_train_df)
        return X_train, None, encoder


# ── Load data once ────────────────────────────────────────────────────────────
train_df, test_df = cached_load_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Data Overview", "📊 EDA", "⚙️ Feature Engineering",
     "🤖 Model Training", "🔮 Predict"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Data Overview")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Training Set")
        st.write(f"**Shape:** {train_df.shape[0]} rows × {train_df.shape[1]} columns")
        st.dataframe(train_df.head(10), width='stretch')
    with col_right:
        st.subheader("Test Set")
        st.write(f"**Shape:** {test_df.shape[0]} rows × {test_df.shape[1]} columns")
        st.dataframe(test_df.head(10), width='stretch')

    st.subheader("Column Info")
    info_df = pd.DataFrame({
        "dtype": train_df.dtypes.astype(str),
        "non-null": train_df.notnull().sum(),
        "null": train_df.isnull().sum(),
        "unique": train_df.nunique(),
    })
    st.dataframe(info_df, width='stretch')

    st.subheader("Target Variable Balance")
    balance = train_df.rainfall.value_counts(normalize=True).reset_index()
    balance.columns = ["rainfall", "proportion"]
    balance["rainfall"] = balance["rainfall"].map({0: "No Rain (0)", 1: "Rain (1)"})

    col_chart, _ = st.columns([1, 2])
    with col_chart:
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.bar(balance["rainfall"], balance["proportion"], color=["#4c9be8", "#e8854c"])
        ax.set_ylabel("Proportion")
        ax.set_title("Target Variable Balance")
        st.pyplot(fig)

    st.subheader("Descriptive Statistics")
    st.dataframe(train_df.describe(include="all"), width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & Visualizations
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Exploratory Data Analysis")

    # ── Box plots per target class ────────────────────────────────────────────
    st.subheader("Feature Distributions by Rainfall Class")
    cols = ["day", "cloud", "humidity", "sunshine", "pressure",
            "temperature", "windspeed", "dewpoint", "mintemp"]
    selected_features = st.multiselect(
        "Select features to plot", cols, default=["cloud", "humidity", "sunshine"]
    )
    if selected_features:
        n = len(selected_features)
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for i, col in enumerate(selected_features):
            sns.boxplot(x="rainfall", y=col, data=train_df, ax=axes[i])
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Mean monthly rainfall ─────────────────────────────────────────────────
    st.subheader("Mean Monthly Rainfall")
    df_month = train_df.copy()
    df_month["month"] = pd.to_datetime(df_month["day"], format="%j").dt.month
    avg_rain = df_month.groupby("month")["rainfall"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    avg_rain.plot(ax=ax, marker="o")
    ax.set(
        title="Mean Monthly Rainfall",
        xticks=range(1, 13),
        xlabel="Month",
        ylabel="Mean Rainfall",
    )
    st.pyplot(fig)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = train_df.corr(numeric_only=True)
    sns.heatmap(corr.round(4), vmin=-1, vmax=1, annot=True, ax=ax, cmap="coolwarm")
    st.pyplot(fig)

    # ── Mutual Information ────────────────────────────────────────────────────
    st.subheader("Mutual Information Scores")
    X_mi = train_df.copy()
    y_mi = X_mi.pop("rainfall")
    disc = X_mi.select_dtypes(include=["int64"])
    mi = get_mutual_info_scores(X_mi, y_mi, disc)
    mi_df = pd.DataFrame(mi, index=X_mi.columns, columns=["MI Score"]).sort_values(
        "MI Score", ascending=False
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(y=mi_df.index, x=mi_df["MI Score"], orient="h", ax=ax)
    ax.set_title("Mutual Information Scores")
    st.pyplot(fig)

    # ── Pairplot (expensive) ──────────────────────────────────────────────────
    st.subheader("Pairplot of Top Features")
    if st.button("Generate Pairplot (may take a moment)"):
        top = ["humidity", "cloud", "sunshine", "windspeed"]
        pp_data = X_mi[top].join(y_mi)
        fig_pp = sns.pairplot(data=pp_data, hue="rainfall")
        st.pyplot(fig_pp.figure)

    # ── Clustering elbow / silhouette ─────────────────────────────────────────
    st.subheader("KMeans Clustering Analysis")
    if st.button("Run Elbow & Silhouette Analysis"):
        X_clust = train_df.copy()
        X_clust.drop(columns=["day"], inplace=True)
        y_clust = X_clust.pop("rainfall")

        inertias, sil_scores = [], []
        k_range = range(2, 11)
        for k in k_range:
            Xn, Xcd, iner = cluster_labels(X_clust, k)
            inertias.append(iner)
            sil_scores.append(metrics.silhouette_score(X_clust, Xn.cluster.values))

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.plot(list(k_range), inertias, marker="o")
            ax.set(title="Elbow Plot", xlabel="k", ylabel="Inertia")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots()
            ax.plot(list(k_range), sil_scores, marker="o", color="orange")
            ax.set(title="Silhouette Score", xlabel="k", ylabel="Score")
            st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Feature Engineering")

    X_eng, _, _ = cached_create_features(train_df)
    y_eng = train_df.loc[:, "rainfall"]

    st.subheader("Engineered Features")
    st.write(f"Shape: {X_eng.shape}")
    st.dataframe(X_eng.head(50), width='stretch')

    st.subheader("Correlation with Target")
    corr_target = X_eng.corrwith(y_eng).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_target.plot.barh(ax=ax)
    ax.set_title("Feature Correlation with Rainfall")
    ax.axvline(0, color="black", linewidth=0.5)
    st.pyplot(fig)

    st.subheader("Variance Inflation Factor (VIF)")
    vif_cols = ["cloud", "humidity", "sunshine", "day_encoded", "dewpoint_temperature"]
    vif_cols = [c for c in vif_cols if c in X_eng.columns]
    if vif_cols:
        vif_data = add_constant(X_eng[vif_cols])
        vif_df = pd.DataFrame({
            "Feature": vif_data.columns,
            "VIF": [variance_inflation_factor(vif_data.values, i)
                    for i in range(vif_data.shape[1])],
        })
        st.dataframe(vif_df, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Training & Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Training & Comparison")

    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}

    # Prepare splits
    X_tr_full, _, encoder_obj = cached_create_features(train_df)
    y_tr_full = train_df.loc[:, "rainfall"]
    tr_x, vl_x, tr_y, vl_y = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, random_state=0, stratify=y_tr_full
    )

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # ── Sidebar hyper-parameter controls ──────────────────────────────────────
    st.sidebar.header("Model Hyperparameters")

    # Logistic Regression
    st.sidebar.subheader("Logistic Regression")
    lr_C = st.sidebar.multiselect(
        "C values", [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        default=[0.01, 0.1, 1, 10],
    )

    # Decision Tree
    st.sidebar.subheader("Decision Tree")
    dt_max_depth = st.sidebar.multiselect(
        "max_depth", [4, 6, 8, 12, "None"], default=[4, 8, 12]
    )
    dt_min_leaf = st.sidebar.multiselect(
        "min_samples_leaf (DT)", [2, 5, 6], default=[2, 5]
    )
    dt_min_split = st.sidebar.multiselect(
        "min_samples_split (DT)", [2, 4, 6], default=[2, 4]
    )

    # Random Forest
    st.sidebar.subheader("Random Forest")
    rf_n_est = st.sidebar.multiselect(
        "n_estimators (RF)", [100, 300, 500], default=[300]
    )
    rf_max_depth = st.sidebar.multiselect(
        "max_depth (RF)", [3, 5, "None"], default=[5, "None"]
    )
    rf_max_samples = st.sidebar.multiselect(
        "max_samples (RF)", [0.7, 1.0], default=[0.7, 1.0]
    )
    rf_min_leaf = st.sidebar.multiselect(
        "min_samples_leaf (RF)", [1, 2, 3], default=[1, 2]
    )

    # XGBoost
    st.sidebar.subheader("XGBoost")
    xgb_max_depth = st.sidebar.multiselect(
        "max_depth (XGB)", [1, 2, 4, 6], default=[1, 2, 4]
    )
    xgb_lr = st.sidebar.multiselect(
        "learning_rate (XGB)", [0.01, 0.05, 0.1], default=[0.01]
    )
    xgb_n_est = st.sidebar.multiselect(
        "n_estimators (XGB)", [300, 500, 800], default=[500]
    )
    xgb_subsample = st.sidebar.multiselect(
        "subsample (XGB)", [0.7, 0.8, 1.0], default=[0.7]
    )

    # ── Model selection & training ────────────────────────────────────────────
    model_choices = st.multiselect(
        "Select models to train",
        ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
        default=["Logistic Regression"],
    )

    def _convert_depth(vals):
        """Convert sidebar depth selections (may contain 'None' string)."""
        return [None if v == "None" else int(v) for v in vals]

    if st.button("🚀 Train Selected Models", type="primary"):
        # ── Logistic Regression ───────────────────────────────────────────────
        if "Logistic Regression" in model_choices:
            with st.spinner("Training Logistic Regression …"):
                x_tr_s = (tr_x - tr_x.mean()) / tr_x.std()
                x_vl_s = (vl_x - vl_x.mean()) / vl_x.std()
                lr_model = LogisticRegression(
                    l1_ratio=1, solver="liblinear", max_iter=2000, random_state=0
                )
                lr_cv = GridSearchCV(
                    lr_model,
                    param_grid={"C": lr_C if lr_C else [1]},
                    scoring=scoring, cv=5, refit="roc_auc",
                )
                lr_cv.fit(x_tr_s, tr_y)
                st.session_state.trained_models["Logistic Regression"] = {
                    "cv": lr_cv,
                    "cv_results": make_results("Logistic Regression", lr_cv, "auc"),
                    "val_results": get_scores("Logistic Regression", lr_cv, x_vl_s, vl_y),
                    "X_val": x_vl_s,
                    "y_val": vl_y,
                    "scaled": True,
                    "tr_mean": tr_x.mean(),
                    "tr_std": tr_x.std(),
                }
            st.success("Logistic Regression trained ✓")

        # ── Decision Tree ─────────────────────────────────────────────────────
        if "Decision Tree" in model_choices:
            with st.spinner("Training Decision Tree …"):
                dt_model = DecisionTreeClassifier(random_state=0)
                dt_params = {
                    "max_depth": _convert_depth(dt_max_depth) if dt_max_depth else [None],
                    "min_samples_leaf": [int(v) for v in dt_min_leaf] if dt_min_leaf else [2],
                    "min_samples_split": [int(v) for v in dt_min_split] if dt_min_split else [2],
                }
                dt_cv = GridSearchCV(
                    dt_model, dt_params, scoring=scoring, refit="roc_auc", cv=5
                )
                dt_cv.fit(tr_x, tr_y)
                st.session_state.trained_models["Decision Tree"] = {
                    "cv": dt_cv,
                    "cv_results": make_results("Decision Tree", dt_cv, "auc"),
                    "val_results": get_scores("Decision Tree", dt_cv, vl_x, vl_y),
                    "X_val": vl_x,
                    "y_val": vl_y,
                    "scaled": False,
                }
            st.success("Decision Tree trained ✓")

        # ── Random Forest ─────────────────────────────────────────────────────
        if "Random Forest" in model_choices:
            with st.spinner("Training Random Forest …"):
                rf_model = RandomForestClassifier(random_state=0)
                rf_params = {
                    "max_depth": _convert_depth(rf_max_depth) if rf_max_depth else [None],
                    "n_estimators": [int(v) for v in rf_n_est] if rf_n_est else [300],
                    "max_samples": [float(v) for v in rf_max_samples] if rf_max_samples else [1.0],
                    "min_samples_leaf": [int(v) for v in rf_min_leaf] if rf_min_leaf else [1],
                    "max_features": [1.0],
                    "min_samples_split": [2, 3],
                }
                rf_cv = GridSearchCV(
                    rf_model, rf_params, scoring=scoring, refit="roc_auc", cv=5, n_jobs=-1
                )
                rf_cv.fit(tr_x, tr_y)
                st.session_state.trained_models["Random Forest"] = {
                    "cv": rf_cv,
                    "cv_results": make_results("Random Forest", rf_cv, "auc"),
                    "val_results": get_scores("Random Forest", rf_cv, vl_x, vl_y),
                    "X_val": vl_x,
                    "y_val": vl_y,
                    "scaled": False,
                }
            st.success("Random Forest trained ✓")

        # ── XGBoost ───────────────────────────────────────────────────────────
        if "XGBoost" in model_choices:
            with st.spinner("Training XGBoost …"):
                xgb_model = XGBClassifier(
                    random_state=0, objective="binary:logistic", verbosity=0
                )
                xgb_params = {
                    "max_depth": [int(v) for v in xgb_max_depth] if xgb_max_depth else [2],
                    "learning_rate": [float(v) for v in xgb_lr] if xgb_lr else [0.01],
                    "n_estimators": [int(v) for v in xgb_n_est] if xgb_n_est else [500],
                    "subsample": [float(v) for v in xgb_subsample] if xgb_subsample else [0.7],
                    "colsample_bytree": [0.7],
                    "min_child_weight": [3],
                }
                xgb_cv = GridSearchCV(
                    xgb_model, xgb_params, scoring=scoring, refit="roc_auc", cv=5
                )
                xgb_cv.fit(tr_x, tr_y)
                st.session_state.trained_models["XGBoost"] = {
                    "cv": xgb_cv,
                    "cv_results": make_results("XGBoost", xgb_cv, "auc"),
                    "val_results": get_scores("XGBoost", xgb_cv, vl_x, vl_y),
                    "X_val": vl_x,
                    "y_val": vl_y,
                    "scaled": False,
                }
            st.success("XGBoost trained ✓")

    # ── Display results for each trained model ────────────────────────────────
    if st.session_state.trained_models:
        for name, info in st.session_state.trained_models.items():
            st.divider()
            st.subheader(name)

            best = info["cv"].best_estimator_
            st.write("**Best Parameters:**", info["cv"].best_params_)

            c1, c2 = st.columns(2)
            with c1:
                st.write("**Cross-Validation Results**")
                st.dataframe(info["cv_results"], width='stretch')
            with c2:
                st.write("**Validation Results**")
                st.dataframe(info["val_results"], width='stretch')

            c3, c4 = st.columns(2)
            with c3:
                # ROC Curve
                fig, ax = plt.subplots()
                metrics.RocCurveDisplay.from_estimator(best, info["X_val"], info["y_val"], ax=ax)
                ax.set_title(f"ROC Curve — {name}")
                st.pyplot(fig)
            with c4:
                # Confusion Matrix
                preds = best.predict(info["X_val"])
                cm = metrics.confusion_matrix(info["y_val"], preds, labels=best.classes_)
                fig, ax = plt.subplots()
                disp = metrics.ConfusionMatrixDisplay(cm, display_labels=best.classes_)
                disp.plot(ax=ax, values_format="")
                ax.set_title(f"Confusion Matrix — {name}")
                st.pyplot(fig)

            # Feature importances (tree-based models)
            if hasattr(best, "feature_importances_"):
                fi = pd.DataFrame(
                    best.feature_importances_,
                    index=best.feature_names_in_,
                    columns=["importance"],
                ).sort_values("importance", ascending=False)
                st.write("**Feature Importances**")
                fig, ax = plt.subplots(figsize=(8, 4))
                fi.plot.barh(ax=ax)
                ax.set_title(f"Feature Importances — {name}")
                ax.invert_yaxis()
                st.pyplot(fig)

            # Coefficients (logistic regression)
            if hasattr(best, "coef_"):
                coefs = pd.DataFrame(
                    best.coef_[0],
                    index=best.feature_names_in_,
                    columns=["coefficient"],
                ).sort_values("coefficient", ascending=False)
                st.write("**Model Coefficients**")
                st.dataframe(coefs, width='stretch')

        # ── Combined comparison table ─────────────────────────────────────────
        st.divider()
        st.subheader("📊 Model Comparison")
        comparison = pd.concat(
            [info["val_results"] for info in st.session_state.trained_models.values()],
            ignore_index=True,
        )
        st.dataframe(
            comparison.style.highlight_max(
                subset=["precision", "recall", "F1", "accuracy", "AUC"],
                color="#90EE90",
            ),
            width='stretch',
        )

        # Grouped bar chart
        comp_melted = comparison.melt(id_vars="model", var_name="metric", value_name="score")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=comp_melted, x="metric", y="score", hue="model", ax=ax)
        ax.set_title("Model Comparison")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Predict
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Make a Prediction")

    if not st.session_state.get("trained_models"):
        st.info("Train at least one model in the **Model Training** tab first.")
    else:
        model_name = st.selectbox(
            "Select model for prediction",
            list(st.session_state.trained_models.keys()),
        )

        st.subheader("Input Weather Features")

        col1, col2, col3 = st.columns(3)
        with col1:
            inp_day = st.number_input("Day of year", min_value=1, max_value=365, value=100)
            inp_pressure = st.number_input("Pressure", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
            inp_maxtemp = st.number_input("Max Temp", min_value=-10.0, max_value=55.0, value=25.0, step=0.1)
            inp_temperature = st.number_input("Temperature", min_value=-10.0, max_value=55.0, value=20.0, step=0.1)
        with col2:
            inp_mintemp = st.number_input("Min Temp", min_value=-10.0, max_value=55.0, value=15.0, step=0.1)
            inp_dewpoint = st.number_input("Dewpoint", min_value=-20.0, max_value=40.0, value=12.0, step=0.1)
            inp_humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
            inp_cloud = st.number_input("Cloud", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
        with col3:
            inp_sunshine = st.number_input("Sunshine", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            inp_winddirection = st.number_input("Wind Direction", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
            inp_windspeed = st.number_input("Wind Speed", min_value=0.0, max_value=130.0, value=20.0, step=0.1)

        if st.button("🔮 Predict", type="primary"):
            # Build a single-row dataframe matching the test set schema
            input_row = pd.DataFrame([{
                "day": inp_day,
                "pressure": inp_pressure,
                "maxtemp": inp_maxtemp,
                "temperature": inp_temperature,
                "mintemp": inp_mintemp,
                "dewpoint": inp_dewpoint,
                "humidity": inp_humidity,
                "cloud": inp_cloud,
                "sunshine": inp_sunshine,
                "winddirection": inp_winddirection,
                "windspeed": inp_windspeed,
            }])
            input_row.index.name = "id"
            input_row.index = input_row.index + test_df.index.max() + 1  # unique index

            # Run full feature pipeline (append to test set so encoder works)
            combined_test = pd.concat([test_df, input_row])
            try:
                _, X_pred_all, _ = create_features(train_df, combined_test)
                X_pred = X_pred_all.iloc[[-1]]  # last row = our input

                model_info = st.session_state.trained_models[model_name]
                best_model = model_info["cv"].best_estimator_

                # Scale if needed (Logistic Regression)
                if model_info.get("scaled"):
                    X_pred = (X_pred - model_info["tr_mean"]) / model_info["tr_std"]

                pred = best_model.predict(X_pred)[0]
                prob = best_model.predict_proba(X_pred)[0]

                st.divider()
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Prediction", "🌧️ Rain" if pred == 1 else "☀️ No Rain")
                with r2:
                    st.metric("Rain Probability", f"{prob[1]:.1%}")
                with r3:
                    st.metric("No-Rain Probability", f"{prob[0]:.1%}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
