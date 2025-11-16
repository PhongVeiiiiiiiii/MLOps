import os
import shutil
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# --- MLflow experiment ---
mlflow.set_experiment("Tabular_Classification_Experiment")

# --- Tạo dữ liệu tabular cải tiến ---
X, y = make_classification(
    n_samples=3000,       # tăng số mẫu
    n_features=10,
    n_informative=8,
    n_redundant=0,        # bỏ feature dư thừa
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Folder lưu model ---
best_model_path = "models/best_model"
os.makedirs("models", exist_ok=True)
if os.path.exists(best_model_path):
    shutil.rmtree(best_model_path)

best_acc = 0
best_model = None
best_model_name = ""

# --- Tuning RandomForest ---
rf_configs = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 150, "max_depth": None}
]

for i, cfg in enumerate(rf_configs):
    with mlflow.start_run(run_name=f"RandomForest_run_{i+1}"):
        model = RandomForestClassifier(**cfg, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(cfg)
        mlflow.log_metric("accuracy", acc)

        input_example = np.zeros((1, X_train.shape[1]))
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        print(f"[RandomForest run {i+1}] accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = f"RandomForest_{i+1}"

# --- Tuning Logistic Regression ---
lr_configs = [
    {"C": 0.1, "solver": "lbfgs"},
    {"C": 1.0, "solver": "lbfgs"},
    {"C": 10.0, "solver": "lbfgs"}
]

for i, cfg in enumerate(lr_configs):
    with mlflow.start_run(run_name=f"LogisticRegression_run_{i+1}"):
        lr_model = LogisticRegression(max_iter=1000, **cfg)
        lr_model.fit(X_train, y_train)
        preds = lr_model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(cfg)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(lr_model, "model", input_example=input_example)

        print(f"[LogisticRegression run {i+1}] accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = lr_model
            best_model_name = f"LogisticRegression_{i+1}"

# --- Lưu model tốt nhất ---
mlflow.sklearn.save_model(best_model, best_model_path, input_example=input_example)
print(f"Model tốt nhất: {best_model_name}, accuracy: {best_acc:.4f}")
print(f"Đã lưu model tốt nhất tại: {best_model_path}")
