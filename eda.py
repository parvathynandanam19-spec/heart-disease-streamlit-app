
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.style.use("ggplot")
sns.set_theme()

def load_cleaned_dataset(filepath="heart_cleaned.csv"):
    df = pd.read_csv(filepath)
    return df

def basic_overview(df):
    print("Dataset Shape:", df.shape)
    print("\nColumn Information:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

def target_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="target", data=df)
    plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def feature_distributions(df):
    numerical_features = ["age", "chol", "trestbps", "thalach", "oldpeak"]
    
    for feature in numerical_features:
        plt.figure(figsize=(7,4))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"{feature.capitalize()} Distribution")
        plt.show()

def boxplots_by_target(df):
    features = ["age", "thalach", "trestbps", "chol", "oldpeak"]
    
    for feature in features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="target", y=feature, data=df)
        plt.title(f"{feature.capitalize()} vs Heart Disease")
        plt.show()

def categorical_plots(df):
    cat_features = ["cp", "sex", "fbs", "exang", "slope", "ca", "thal"]
    
    for feature in cat_features:
        plt.figure(figsize=(7,4))
        sns.countplot(x=feature, hue="target", data=df)
        plt.title(f"{feature.upper()} by Heart Disease")
        plt.show()

def pairplot(df):
    sns.pairplot(df[["age", "chol", "thalach", "trestbps", "target"]], hue="target")
    plt.show()

def perform_eda():
    df = load_cleaned_dataset()
    
    print("\nBASIC DATASET OVERVIEW")
    basic_overview(df)
    
    print("\nTARGET VARIABLE DISTRIBUTION")
    target_distribution(df)

    print("\nCORRELATION HEATMAP")
    correlation_heatmap(df)

    print("\nNUMERICAL FEATURE DISTRIBUTIONS")
    feature_distributions(df)

    print("\nBOX PLOTS BY TARGET")
    boxplots_by_target(df)

    print("\nCATEGORICAL FEATURE PLOTS")
    categorical_plots(df)

    print("\nPAIRPLOT OF SELECT FEATURES")
    pairplot(df)

if __name__ == "__main__":
    perform_eda()
