import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "klasyfikator-nowotworu"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("lab3-rejestr-modeli")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

params = {
    "n_estimators": 150,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42,
}

with mlflow.start_run(run_name="model-gradient-boosting"):
    mlflow.log_params(params)

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))  # type: ignore
    mlflow.log_metric("f1_macro", f1_score(y_test, y_pred, average="macro"))  # type: ignore

    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(  # type: ignore
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name=MODEL_NAME,
    )
