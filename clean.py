import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from load_dataset import load_cleveland_dataset

def clean_cleveland_dataset(save_path="heart_cleaned.csv"):

    df = load_cleveland_dataset()

    # Remove header-like row accidentally read as data
    df = df[df['age'] != 'age']

    # Drop missing values
    df.dropna(inplace=True)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric)

    # Convert target to binary
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop("target", axis=1))

    df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])
    df_scaled["target"] = df["target"].values

    # Save cleaned dataset
    df_scaled.to_csv(save_path, index=False)
    print(f"Cleaned dataset saved at: {save_path}")

    return df_scaled


if __name__ == "__main__":
    cleaned = clean_cleveland_dataset()
    print("Cleaning completed!")
    print(cleaned.head())
