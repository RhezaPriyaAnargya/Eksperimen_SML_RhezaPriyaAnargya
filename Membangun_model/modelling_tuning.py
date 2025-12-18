import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# =========================
# INIT DAGS HUB (WAJIB)
# =========================
dagshub.init(
    repo_owner="RhezaPriyaAnargya",      
    repo_name="telco-churn-mlflow",      
    mlflow=True
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("namadataset_preprocessing/telco_preprocessed.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Telco-Churn-Advanced")

# =========================
# HYPERPARAMETER TUNING
# =========================
C_values = [0.1, 1.0, 10.0]

os.makedirs("artifacts", exist_ok=True)

for C in C_values:
    with mlflow.start_run():
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        # -------- MANUAL LOGGING --------
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        mlflow.sklearn.log_model(model, "model")

        # =========================
        # ARTEFAK TAMBAHAN (WAJIB ≥ 2)
        # =========================

        # 1️ Confusion Matrix Image
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.savefig(f"artifacts/cm_C_{C}.png")
        plt.close()

        mlflow.log_artifact(f"artifacts/cm_C_{C}.png")

        # 2️ Summary Text
        summary_text = f"""
        C = {C}
        Accuracy = {acc}
        Precision = {prec}
        Recall = {rec}
        """
        with open(f"artifacts/summary_C_{C}.txt", "w") as f:
            f.write(summary_text)

        mlflow.log_artifact(f"artifacts/summary_C_{C}.txt")
