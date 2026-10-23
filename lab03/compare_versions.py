from mlflow import MlflowClient

MODEL_NAME = "klasyfikator-nowotworu"
VERSIONS = ["1", "2"]

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

results = {}
for version in VERSIONS:
    model_version = client.get_model_version(MODEL_NAME, version)
    run = client.get_run(model_version.run_id)  # type: ignore
    results[version] = {
        "accuracy": run.data.metrics["accuracy"],
        "f1_macro": run.data.metrics["f1_macro"],
    }

print(f"Porównanie wersji modelu {MODEL_NAME}")
print(f"{'Wersja':<8}{'accuracy':<12}{'f1_macro':<12}")
for version, metrics in results.items():
    print(f"{version:<8}{metrics['accuracy']:<12.4f}{metrics['f1_macro']:<12.4f}")

best_version = max(results, key=lambda v: results[v]["f1_macro"])
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=best_version,
)

print(f"Alias champion przypięty do wersji {best_version}")
