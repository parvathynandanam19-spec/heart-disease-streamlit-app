
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
import joblib
import shap
import os
from lime.lime_tabular import LimeTabularExplainer
from io import BytesIO



warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Try to import user feature_extraction if present
try:
    from train1 import feature_extraction
    HAS_FEATURE_EXTRACTION = True
except Exception:
    HAS_FEATURE_EXTRACTION = False

st.set_page_config(page_title="Heart Disease ML Dashboard", layout="wide")
st.title("Heart Disease Data Evaluation Dashboard")

# ---------------------------
# Sidebar: settings & upload
# ---------------------------
st.sidebar.header("1) Dataset")
dataset_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("2) Preprocessing options")
apply_feature_extraction = st.sidebar.button("Apply feature_extraction() (if available)")
apply_fill_missing = st.sidebar.checkbox("Auto-fill missing (numeric→median, cat→mode)", value=True)
apply_skew = st.sidebar.checkbox("Apply skewness correction (Yeo-Johnson)", value=False)
skew_cols = st.sidebar.text_input("Skewness columns (comma-separated). Default 'oldpeak'", value="oldpeak")

st.sidebar.markdown("---")
st.sidebar.header("3) Balancing (auto suggestion)")
auto_balance = st.sidebar.checkbox("Auto-apply SMOTE when needed", value=False)
balance_threshold = st.sidebar.slider("If minority proportion ≤", min_value=0.01, max_value=0.49, value=0.25, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("4) Modeling & Tuning")
models_to_train = st.sidebar.multiselect("Models to train", ["Logistic Regression", "Random Forest", "KNN", "Decision Tree"],
                                         default=["Logistic Regression", "Random Forest"])
test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
enable_tuning = st.sidebar.checkbox("Enable hyperparameter tuning (Grid/Randomized)", value=False)
tuning_method = st.sidebar.selectbox("Tuning method", ["GridSearchCV", "RandomizedSearchCV"])
n_iter = st.sidebar.number_input("RandomizedSearch n_iter", value=20, min_value=5, max_value=200, step=5)
cv_folds = st.sidebar.number_input("CV folds", value=5, min_value=2, max_value=10, step=1)
scoring_metric = st.sidebar.selectbox("Tuning scoring metric", ["accuracy", "f1", "roc_auc"], index=0)
@st.cache_resource
def load_saved_pipeline():
    if not os.path.exists("model/best_model.pkl"):
        return None, None, None
    model = joblib.load("model/best_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    features = joblib.load("model/feature_columns.pkl")
    return model, scaler, features

st.sidebar.markdown("---")
st.sidebar.header("5) Quick predict")
show_quick_predict = st.sidebar.checkbox("Show quick single-sample prediction", value=False)

# ---------------------------
# Load dataset
# ---------------------------
if dataset_file is None:
    st.info("Upload a CSV dataset to start. Make sure your target column is named 'target'.")
    st.stop()

# df = pd.read_csv(dataset_file)
# Read bytes once so we can reuse the uploaded file safely
file_bytes = dataset_file.getvalue()
df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
st.success("Dataset loaded.")
st.subheader("Dataset preview (first 5 rows)")
st.dataframe(df.head())

if 'target' not in df.columns:
    st.error("No 'target' column found.")
    st.stop()

# Fill missing
if apply_fill_missing:
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object','category']).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])
    st.write("Missing values filled (numeric→median, categorical→mode).")

# Feature extraction if requested
if apply_feature_extraction:
    if HAS_FEATURE_EXTRACTION:
        try:
            df = feature_extraction(df)
            st.success("feature_extraction() applied.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"feature_extraction() raised an error: {e}")
    else:
        st.warning("feature_extraction() not found.")

# Convert low-cardinality numeric to categorical
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64] and df[col].nunique() <= 10 and col != 'target':
        df[col] = df[col].astype('category')

# Skew correction (Yeo-Johnson)
if apply_skew:
    sel_cols = [c.strip() for c in skew_cols.split(",") if c.strip() in df.columns] if skew_cols.strip() else ['oldpeak'] if 'oldpeak' in df.columns else []
    if sel_cols:
        pt = PowerTransformer(method='yeo-johnson')
        for c in sel_cols:
            vals = df[c].astype(float).values.reshape(-1,1)
            df[c+"_yj"] = pt.fit_transform(vals).flatten()
            st.write(f"{c} transformed (Yeo-Johnson)")
    else:
        st.warning("No valid numeric columns selected for skew correction.")

# ---------------------------
# Balancing suggestion & optional SMOTE
# ---------------------------
st.subheader("Target class distribution")
class_counts = df['target'].value_counts()
st.write(class_counts)
needs_balancing = False
if len(class_counts) == 2:
    minority_prop = class_counts.min() / class_counts.sum()
    if minority_prop <= balance_threshold:
        st.warning(f"Minority proportion ({minority_prop:.3f}) ≤ threshold → balancing recommended.")
        needs_balancing = True
    else:
        st.info("Balancing probably not needed.")

df_balanced = df.copy()
if auto_balance and needs_balancing and len(class_counts)==2:
    st.write("Applying SMOTE...")
    X_sm = pd.get_dummies(df.drop('target', axis=1), drop_first=True)
    y_sm = df['target']
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_sm, y_sm)
    df_balanced = pd.concat([pd.DataFrame(X_res, columns=X_sm.columns), pd.Series(y_res, name='target')], axis=1)
    st.success("SMOTE applied. New class counts:")
    st.write(df_balanced['target'].value_counts())

# ---------------------------
# EDA & visualizations
# ---------------------------
if st.checkbox("Show EDA & Feature Importance"):
    st.subheader("Correlation heatmap")
    num_df = df_balanced.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        plt.clf()

    st.subheader("Boxplots of numeric features")
    numeric_cols = [c for c in num_df.columns if c != 'target']
    if numeric_cols:
        col_box = st.selectbox("Select column for boxplot", numeric_cols)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='target', y=col_box, data=df_balanced, ax=ax)
        st.pyplot(fig)
        plt.clf()

    st.subheader("Histograms of numeric features")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df_balanced[col], kde=True, ax=ax)
        st.pyplot(fig)
        plt.clf()

    st.subheader("Bar/Pie charts for categorical features")
    cat_cols = df_balanced.select_dtypes(include=['category','object']).columns.tolist()
    for col in cat_cols:
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        df_balanced[col].value_counts().plot(kind='bar', ax=ax[0], color='skyblue')
        ax[0].set_title(f"Bar plot of {col}")
        df_balanced[col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax[1])
        ax[1].set_ylabel('')
        ax[1].set_title(f"Pie chart of {col}")
        st.pyplot(fig)
        plt.clf()

    st.subheader("Feature importance (Random Forest)")
    X_feat = pd.get_dummies(df_balanced.drop('target', axis=1), drop_first=True)
    y_feat = df_balanced['target'].astype(int)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_feat, y_feat)
    importances = pd.Series(rf.feature_importances_, index=X_feat.columns).sort_values(ascending=False)
    st.write("Top 10 features contributing to heart disease:")
    st.dataframe(importances.head(10))
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=importances.head(10), y=importances.head(10).index, palette="viridis", ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)
    plt.clf()

# ---------------------------
# Prepare data for modeling
# ---------------------------
X_full = pd.get_dummies(df_balanced.drop('target', axis=1), drop_first=True)
y_full = df_balanced['target'].astype(int)
X_full = X_full.fillna(X_full.median())
stratify_arg = y_full if len(y_full.unique())==2 else None
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=int(random_state), stratify=stratify_arg)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ---------------------------
# Model training & evaluation
# ---------------------------
model_constructors = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "KNN": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier
}

param_grids = {
    "Logistic Regression": {"C":[0.01,0.1,1,10], "solver":["lbfgs"], "max_iter":[200]},
    "Random Forest": {"n_estimators":[100,200], "max_depth":[None,5,10], "min_samples_split":[2,5]},
    "KNN": {"n_neighbors":[3,5,7], "weights":["uniform","distance"]},
    "Decision Tree": {"max_depth":[None,3,5,10], "min_samples_split":[2,5,10]}
}

if st.button("Train & Evaluate"):
    if not models_to_train:
        st.warning("Select at least one model to train (sidebar).")
    else:
        results = []
        trained_models = {}
        for model_name in models_to_train:
            st.markdown(f"## {model_name}")
            ModelClass = model_constructors[model_name]
            Xtr, Xte = (X_train_scaled, X_test_scaled) if model_name in ["Logistic Regression","KNN"] else (X_train, X_test)
            base_model = ModelClass(random_state=int(random_state)) if model_name not in ["KNN"] else ModelClass()
            chosen_model = base_model

            # Hyperparameter tuning
            if enable_tuning and model_name in param_grids:
                grid = param_grids[model_name]
                try:
                    if tuning_method=="GridSearchCV":
                        search = GridSearchCV(base_model, grid, cv=int(cv_folds), scoring=scoring_metric, n_jobs=-1)
                        search.fit(Xtr, y_train)
                    else:
                        search = RandomizedSearchCV(base_model, grid, n_iter=int(n_iter), cv=int(cv_folds),
                                                    scoring=scoring_metric, n_jobs=-1, random_state=int(random_state))
                        search.fit(Xtr, y_train)
                    chosen_model = search.best_estimator_
                    st.write("Best params:", search.best_params_)
                    st.write(f"Best CV {scoring_metric}: {search.best_score_:.4f}")
                except Exception:
                    st.warning(f"Tuning failed for {model_name}. Using default params.")

            # Fit model
            chosen_model.fit(Xtr, y_train)
            trained_models[model_name] = chosen_model

            y_pred = chosen_model.predict(Xte)
            y_proba = chosen_model.predict_proba(Xte)[:,1] if hasattr(chosen_model,"predict_proba") and len(np.unique(y_test))==2 else None

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test))==2 else 'macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test))==2 else 'macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test))==2 else 'macro', zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

            results.append({
                "Model": model_name, "Accuracy": round(acc,4), "Precision": round(prec,4),
                "Recall": round(rec,4), "F1-score": round(f1,4), "ROC AUC": round(roc_auc,4) if roc_auc else None
            })

            st.write(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}" + (f", ROC AUC: {roc_auc:.3f}" if roc_auc else ""))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{model_name} Confusion Matrix")
            st.pyplot(fig_cm)
            plt.clf()

        # Show results table
        if results:
            comp_df = pd.DataFrame(results).set_index("Model")
            st.subheader("Model comparison table")
            st.dataframe(comp_df)
            csv = comp_df.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button("Download comparison CSV", data=csv, file_name="model_comparison.csv", mime="text/csv")
# =====================================================
# QUICK PREDICTION + SHAP + LIME (DEPLOYED MODEL)
# =====================================================
if show_quick_predict:
    st.markdown("---")
    st.header("Single Patient Prediction & Explainability")

    model, saved_scaler, saved_features = load_saved_pipeline()

    if model is None:
        st.error("No trained model found. Please run train.py first.")
        st.stop()

    # Load original dataset for reference & LIME background
    # raw_df = pd.read_csv(dataset_file)
    raw_df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    raw_df = pd.read_csv(BytesIO(file_bytes))


    raw_df = raw_df.drop("target", axis=1)

    # Apply same feature extraction if used
    if HAS_FEATURE_EXTRACTION:
        try:
            raw_df = feature_extraction(raw_df)
        except:
            pass

    raw_df = raw_df[saved_features]

    st.subheader("Patient Clinical Inputs")

    inputs = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(saved_features):
        with col1 if i % 2 == 0 else col2:
            default_val = float(raw_df[feature].median())
            inputs[feature] = st.number_input(
                label=feature,
                value=default_val,
                step=0.1
            )

    input_df = pd.DataFrame([inputs])
    input_scaled = saved_scaler.transform(input_df)

    # ---------------------------
    # Prediction
    # ---------------------------
    if st.button("Predict Heart Disease Risk"):
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"High Risk of Heart Disease\n\nProbability: {prob:.2f}")
        else:
            st.success(f" Low Risk of Heart Disease\n\nProbability: {prob:.2f}")

        # ---------------------------
        # SHAP Explanation (LOCAL)
        # ---------------------------
        st.subheader("SHAP Explanation (Local)")

        try:
            background = saved_scaler.transform(raw_df.sample(min(100, len(raw_df))))
            explainer = shap.Explainer(model, background)
            shap_values = explainer(input_scaled)

            st.pyplot(
                shap.plots.waterfall(shap_values[0], show=False),
                clear_figure=True
            )
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # ---------------------------
        # LIME Explanation (LOCAL)
        # ---------------------------
        st.subheader("LIME Explanation (Local)")

        try:
            lime_explainer = LimeTabularExplainer(
                training_data=saved_scaler.transform(raw_df),
                feature_names=saved_features,
                class_names=["No Disease", "Disease"],
                mode="classification"
            )

            lime_exp = lime_explainer.explain_instance(
                input_scaled[0],
                model.predict_proba,
                num_features=8
            )

            st.components.v1.html(lime_exp.as_html(), height=800)

        except Exception as e:
            st.warning(f"LIME explanation unavailable: {e}")
