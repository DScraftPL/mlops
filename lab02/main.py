import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
from optuna.trial import TrialState
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

print(optuna.__version__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50
SEED = 42


X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline = RandomForestClassifier(random_state=42)

baseline_cv = cross_val_score(baseline, X_train, y_train, cv=5, scoring="accuracy")
print(
    f"Baseline, dokladnosc CV: {baseline_cv.mean():.4f} (odch. std. {baseline_cv.std():.4f})"
)

baseline.fit(X_train, y_train)
baseline_test_acc = baseline.score(X_test, y_test)
print(f"Baseline, dokladnosc na zbiorze testowym: {baseline_test_acc:.4f}")


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),

        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return score


study = optuna.create_study(
    direction="maximize",
    study_name="rf_tuning",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS)

print("Najlepsza wartosc funkcji celu (dokladnosc CV):", study.best_value)
print("Najlepsza konfiguracja hiperparametrow:", study.best_params)

best_model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
tuned_test_acc = best_model.score(X_test, y_test)  # zbior testowy uzyty JEDEN raz

print(f"Model strojony, dokladnosc na zbiorze testowym: {tuned_test_acc:.4f}")
print(f"Model bazowy, dokladnosc na zbiorze testowym: {baseline_test_acc:.4f}")

tabela = pd.DataFrame(
    {
        "Model": ["Bazowy", "Strojony (Optuna)"],
        "Dokladnosc CV": [baseline_cv.mean(), study.best_value],
        "Dokladnosc testowa": [baseline_test_acc, tuned_test_acc],
    }
)
print(tabela.round(4).to_string(index=False))

df_trials = study.trials_dataframe()
print(df_trials.head())


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)


def objective_pruning(trial):
    params = {
        "n_estimators": 300,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_tr, y_tr)

    for step, y_pred in enumerate(model.staged_predict(X_val)):
        acc = accuracy_score(y_val, y_pred)
        trial.report(acc, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return acc


study_pruned = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
)
study_pruned.optimize(objective_pruning, n_trials=40)  # type: ignore

pruned = [t for t in study_pruned.trials if t.state == TrialState.PRUNED]
complete = [t for t in study_pruned.trials if t.state == TrialState.COMPLETE]
print(f"Proby ukonczone: {len(complete)}, proby przyciete: {len(pruned)}")

from optuna.visualization import plot_optimization_history, plot_param_importances

fig1 = plot_optimization_history(study)
fig1.show()

fig2 = plot_param_importances(study)
fig2.show()

study_db = optuna.create_study(
    direction="maximize",
    study_name="rf_tuning_sqlite",
    storage="sqlite:///optuna_rf.db",
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study_db.optimize(objective, n_trials=20, n_jobs=2)
print("Liczba prob w bazie:", len(study_db.trials))
print("Najlepsza wartosc:", study_db.best_value)

study_tpe = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
)
study_rand = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.RandomSampler(seed=SEED)
)
study_tpe.optimize(objective, n_trials=N_TRIALS)
study_rand.optimize(objective, n_trials=N_TRIALS)

print(f"TPE:    best CV = {study_tpe.best_value:.4f}")
print(f"Random: best CV = {study_rand.best_value:.4f}")


def best_so_far(s):
    return np.maximum.accumulate([t.value for t in s.trials])


fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=best_so_far(study_tpe), name="TPE"))
fig3.add_trace(go.Scatter(y=best_so_far(study_rand), name="Random"))
fig3.update_layout(
    title="TPE vs Random: najlepszy wynik CV",
    xaxis_title="Numer proby",
    yaxis_title="Dokladnosc CV",
)
fig3.show()
