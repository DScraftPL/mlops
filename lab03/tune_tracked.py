import optuna
from optuna_integration.mlflow import MLflowCallback
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

TRACKING_URI = "http://127.0.0.1:5000"

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)  # type: ignore
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


mlflow_callback = MLflowCallback(
    tracking_uri=TRACKING_URI,
    metric_name="cv_accuracy",
)

study = optuna.create_study(
    study_name="lab3-optuna-mlflow",
    direction="maximize",
)
study.optimize(objective, n_trials=30, callbacks=[mlflow_callback])

print(f"Najlepsza wartosc: {study.best_value:.4f}")
print(f"Najlepsze parametry: {study.best_params}")
