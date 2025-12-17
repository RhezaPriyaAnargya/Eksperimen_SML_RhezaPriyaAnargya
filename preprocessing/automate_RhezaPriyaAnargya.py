import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop(columns=["customerID"], inplace=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes":1, "No":0})

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    df_out = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    )
    df_out["target"] = y.values
    return df_out

def save(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_data("namadataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_clean = preprocess(df)
    save(df_clean, "namadataset_preprocessing/telco_preprocessed.csv")
