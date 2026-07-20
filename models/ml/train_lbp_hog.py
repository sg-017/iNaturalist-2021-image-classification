import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

BASE_DIR = Path(__file__).parent
FEATURE_DIR = BASE_DIR / "outputs" / "features"
MODEL_DIR = BASE_DIR / "outputs" / "models"
RESULT_DIR = BASE_DIR / "outputs" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def load_features(split):
    feature_path = FEATURE_DIR / f"{split}_features.npz"

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {feature_path}")

    data = np.load(feature_path)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int32)

    print(f"Loaded {split}: features={features.shape}, labels={labels.shape}")

    return features, labels


def calculate_metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1)
    }

    return metrics


def evaluate_model(model, features, labels, split):
    start_time = perf_counter()
    predictions = model.predict(features)
    prediction_time = perf_counter() - start_time

    metrics = calculate_metrics(labels, predictions)
    if hasattr(model,"decision_function"):
        scores = model.decision_function(features)
    elif hasattr(model,"predict_proba"):
        scores = model.predict_proba(features)
    else:
        scores = None
    if scores is not None:
        top5 = np.argsort(scores,axis=1)[:,-5:]
        if hasattr(model,"classes_"):
            classes = model.classes_
        else:
            classes = model.named_steps["classifier"].classes_
        top5_accuracy = np.mean(
            [label in classes[top5[i]] 
            for i, label in enumerate(labels)]
            )  
    else:
        top5_accuracy = None
    
    metrics["top1_accuracy"] = metrics["accuracy"]
    metrics["top5_accuracy"] = top5_accuracy
   
    metrics["prediction_time_seconds"] = prediction_time

    print(f"\n{split} results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Top1 accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"Top5 accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Prediction time: {prediction_time:.2f} seconds")

    return metrics, predictions


def save_results(
        model_name,
        method,
        feature,
        classifier,
        training_time,
        validation_metrics,
        test_metrics,
        test_labels,
        test_predictions
):
    metrics_path = RESULT_DIR / f"{model_name}_metrics.json"

    results = {
        "model": model_name,
        "method": method,
        "feature": feature,
        "classifier": classifier,
        "num_classes": 500,
        "seed": RANDOM_SEED,

        "overall_accuracy": test_metrics["accuracy"],
        "top1_accuracy": test_metrics["top1_accuracy"],
        "top5_accuracy": test_metrics["top5_accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],

        "training_seconds": training_time,
        "test_seconds": test_metrics["prediction_time_seconds"],

        "validation": validation_metrics,
        "test": test_metrics
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    report = classification_report(
        test_labels,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    report_path = RESULT_DIR / f"{model_name}_classification_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    predictions_path = RESULT_DIR / f"{model_name}_test_predictions.npz"

    np.savez_compressed(
        predictions_path,
        labels=test_labels,
        predictions=test_predictions
    )

    print(f"Saved results to: {RESULT_DIR}")


def train_svm(train_features, train_labels):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SGDClassifier(
            loss="hinge",
            alpha=0.0001,
            max_iter=100,
            tol=1e-3,
            n_jobs=-1,
            random_state=RANDOM_SEED
        ))
    ])

    print("\nTraining LBP + HOG + Linear SVM...")

    start_time = perf_counter()
    model.fit(train_features, train_labels)
    training_time = perf_counter() - start_time

    print(f"SVM training completed in {training_time:.2f} seconds")

    return model, training_time


def train_random_forest(train_features, train_labels):
    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1
    )

    print("\nTraining LBP + HOG + Random Forest...")

    start_time = perf_counter()
    model.fit(train_features, train_labels)
    training_time = perf_counter() - start_time

    print(f"Random Forest training completed in {training_time:.2f} seconds")

    return model, training_time


def run_experiment(
        model_name,
        method,
        feature,
        classifier,
        train_function,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels
):
    model, training_time = train_function(train_features, train_labels)

    validation_metrics, _ = evaluate_model(
        model,
        val_features,
        val_labels,
        "Validation"
    )

    test_metrics, test_predictions = evaluate_model(
        model,
        test_features,
        test_labels,
        "Test"
    )

    model_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    print(f"Saved model to: {model_path}")

    save_results(
        model_name,
        method,
        feature,
        classifier,
        training_time,
        validation_metrics,
        test_metrics,
        test_labels,
        test_predictions
    )


def main():
    train_features, train_labels = load_features("train")
    val_features, val_labels = load_features("val")
    test_features, test_labels = load_features("test")

    run_experiment(
        "lbp_hog_svm",
        "Traditional ML",
        "LBP + HOG",
        "Linear SVM",
        train_svm,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels
    )


   # run_experiment( 
        # "lbp_hog_rf",
       # "Traditional ML",
       # "LBP + HOG",
       # "Random Forest",
        #train_random_forest,
       # train_features,
       # train_labels,
       # val_features,
       # val_labels,
       # test_features,
       # test_labels
   # )


if __name__ == "__main__":
    main()