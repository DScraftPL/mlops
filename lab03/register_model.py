import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "klasyfikator-nowotworu"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("lab3-rejestr-modeli")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

best_params = {
    "n_estimators": 267,
    "max_depth": 10,
    "min_samples_split": 3,
    "random_state": 42,
}

with mlflow.start_run(run_name="model-finalny"):
    mlflow.log_params(best_params)

    model = RandomForestClassifier(**best_params)  # type: ignore
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))  # type: ignore
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average="macro"))  # type: ignore

    signature = infer_signature(X_train, model.predict(X_train))

    info = mlflow.sklearn.log_model(  # type: ignore
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

print(f"Zarejestrowano: {info.model_uri}")
