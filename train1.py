import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import os

# -------------------------------
# Load dataset
# -------------------------------
def load_data(filepath="heart_cleaned.csv"):
    return pd.read_csv(filepath)

# -------------------------------
# Feature Engineering
# -------------------------------
def feature_extraction(df):
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=[20, 40, 55, 70, 90], labels=[0, 1, 2, 3])
    df["age_group"] = df["age_group"].cat.add_categories([-1]).fillna(-1)
    df["chol_risk"] = (df["chol"] > 240).astype(int)
    df["bp_risk"] = (df["trestbps"] > 130).astype(int)
    return df

# -------------------------------
# Balance classes (SAFE)
# -------------------------------
def balance_classes(X, y):
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_imputed, y)

    return X_res, y_res

# -------------------------------
# Feature Selection (PRESERVE NAMES)
# -------------------------------
def select_features(X, y, k=12):
    mi = mutual_info_classif(X, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected_features = mi_series.head(k).index.tolist()
    return X[selected_features], selected_features

# -------------------------------
# Train Models
# -------------------------------
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained

# -------------------------------
# Evaluate Models
# -------------------------------
def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }

        print(f"\n{name} Results:")
        for k, v in results[name].items():
            print(f"{k}: {v:.4f}")

    return results

# -------------------------------
# Save Best Model + Pipeline
# -------------------------------
def save_best_model(models, results, scaler, feature_columns):
    os.makedirs("model", exist_ok=True)

    best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
    best_model = models[best_model_name]

    joblib.dump(best_model, "model/best_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(feature_columns, "model/feature_columns.pkl")

    print(f"\nBest Model Saved: {best_model_name}")

# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Loading dataset...")
    df = load_data()

    print("Applying feature engineering...")
    df = feature_extraction(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    print("Balancing dataset using SMOTE...")
    X_res, y_res = balance_classes(X, y)

    print("Selecting important features...")
    X_sel, selected_features = select_features(X_res, y_res, k=12)
    print("Selected Features:", selected_features)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training models...")
    models = train_models(X_train_scaled, y_train)

    print("Evaluating models...")
    results = evaluate_models(models, X_test_scaled, y_test)

    print("Saving best model and pipeline...")
    save_best_model(models, results, scaler, selected_features)

    print("\n🎉 Training complete. Pipeline ready for UI.")

if __name__ == "__main__":
    main()
