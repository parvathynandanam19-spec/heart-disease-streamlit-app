
import pandas as pd

def load_cleveland_dataset(filepath="D:\Devu\Heartdisease\Heart_Disease_Cleveland.csv"):
   
    # Column names provided by UCI repository
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal', 'target'
    ]

    # Load data
    df = pd.read_csv(filepath, names=columns, na_values='?')

    return df


if __name__ == "__main__":
    df = load_cleveland_dataset()
    print("Raw data loaded successfully!")
    print(df.head())
