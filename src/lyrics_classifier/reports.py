import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score

from .config import TrainingConfig
from .modeling import MODEL_DISPLAY_NAMES, MODEL_NAMES, build_pipeline


TFIDF_EXPERIMENTS = (
    {
        "experiment": "default",
        "tfidf_min_df": 2,
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": (1, 2),
        "tfidf_sublinear_tf": True,
    },
    {
        "experiment": "unigrams_only",
        "tfidf_min_df": 2,
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": (1, 1),
        "tfidf_sublinear_tf": True,
    },
    {
        "experiment": "min_df_1",
        "tfidf_min_df": 1,
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": (1, 2),
        "tfidf_sublinear_tf": True,
    },
    {
        "experiment": "max_features_3000",
        "tfidf_min_df": 2,
        "tfidf_max_features": 3000,
        "tfidf_ngram_range": (1, 2),
        "tfidf_sublinear_tf": True,
    },
    {
        "experiment": "no_sublinear_tf",
        "tfidf_min_df": 2,
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": (1, 2),
        "tfidf_sublinear_tf": False,
    },
)


def save_reports(
    y_test: pd.Series,
    predictions,
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
    cv_scores,
    n_albums: int,
    n_songs: int,
    n_splits: int,
    config: TrainingConfig,
) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "n_albums": int(n_albums),
        "n_songs": int(n_songs),
        "min_songs_per_album": int(config.min_songs_per_album),
        "test_size": float(config.test_size),
        "random_state": int(config.random_state),
        "cv_n_splits": int(n_splits),
        "model_name": config.model_name,
        "tfidf_min_df": int(config.tfidf_min_df),
        "tfidf_max_features": int(config.tfidf_max_features),
        "tfidf_ngram_range": list(config.tfidf_ngram_range),
        "tfidf_sublinear_tf": bool(config.tfidf_sublinear_tf),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_accuracy_scores": [float(score) for score in cv_scores],
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
    }

    with open(config.metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    report = classification_report(y_test, predictions, zero_division=0)
    with open(config.classification_report_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\nZapisano metryki do: {config.metrics_path}")
    print(
        "Zapisano raport klasyfikacji do: "
        f"{config.classification_report_path}"
    )

    return metrics


def save_confusion_matrix(y_test, predictions, labels, config: TrainingConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 14))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    display.plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=False)

    plt.title("Confusion matrix - Lyrics Classifier")
    plt.tight_layout()

    plt.savefig(config.confusion_matrix_path, dpi=200)
    plt.close(fig)

    print(f"Zapisano confusion matrix do: {config.confusion_matrix_path}")


def save_errors_csv(X_test, y_test, predictions, pipeline, config: TrainingConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    errors = pd.DataFrame(
        {
            "lyrics": X_test,
            "true_album": y_test,
            "predicted_album": predictions,
        }
    )

    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        errors["confidence"] = probabilities.max(axis=1)
    else:
        errors["confidence"] = None

    errors = errors[errors["true_album"] != errors["predicted_album"]].copy()

    errors["lyrics_preview"] = (
        errors["lyrics"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.slice(0, 250)
    )

    errors = errors.drop(columns=["lyrics"])
    errors.to_csv(config.errors_path, index=False, encoding="utf-8")

    print(f"Zapisano błędne predykcje do: {config.errors_path}")
    print(f"Liczba błędów w zbiorze testowym: {len(errors)}")

    return len(errors)


def save_error_summary_csv(y_test, predictions, config: TrainingConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    errors = pd.DataFrame(
        {
            "true_album": y_test,
            "predicted_album": predictions,
        }
    )

    errors = errors[errors["true_album"] != errors["predicted_album"]].copy()

    error_summary = (
        errors.groupby(["true_album", "predicted_album"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    error_summary.to_csv(config.error_summary_path, index=False, encoding="utf-8")

    print(f"Zapisano podsumowanie błędów do: {config.error_summary_path}")

    return config.error_summary_path


def save_model_comparison(X, y, X_train, X_test, y_train, y_test, cv, config) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for model_name in MODEL_NAMES:
        pipeline = build_pipeline(
            model_name,
            config.random_state,
            tfidf_min_df=config.tfidf_min_df,
            tfidf_max_features=config.tfidf_max_features,
            tfidf_ngram_range=config.tfidf_ngram_range,
            tfidf_sublinear_tf=config.tfidf_sublinear_tf,
        )

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append(
            {
                "model": model_name,
                "cv_accuracy_mean": cv_scores.mean(),
                "cv_accuracy_std": cv_scores.std(),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_macro_f1": f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "test_weighted_f1": f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        by="test_macro_f1",
        ascending=False,
    )

    comparison.to_csv(config.model_comparison_path, index=False, encoding="utf-8")

    print(f"Zapisano porównanie modeli do: {config.model_comparison_path}")
    print("\nPorównanie modeli:")
    print(comparison)

    return config.model_comparison_path


def save_top_features_csv(pipeline, config: TrainingConfig, top_n: int = 15) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(classifier, "coef_"):
        weights_by_class = classifier.coef_
    elif hasattr(classifier, "feature_log_prob_"):
        weights_by_class = classifier.feature_log_prob_
    else:
        print(
            "Pominięto top features: model nie udostępnia wag cech "
            f"({config.model_name})."
        )
        return config.top_features_path

    rows = []
    for class_index, album in enumerate(classifier.classes_):
        weights = weights_by_class[class_index]
        top_indices = weights.argsort()[::-1][:top_n]

        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "album": album,
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "weight": weights[feature_index],
                }
            )

    top_features = pd.DataFrame(rows)
    top_features.to_csv(config.top_features_path, index=False, encoding="utf-8")

    print(f"Zapisano top features do: {config.top_features_path}")

    return config.top_features_path


def save_tfidf_experiments(X, y, X_train, X_test, y_train, y_test, cv, config) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for experiment in TFIDF_EXPERIMENTS:
        pipeline = build_pipeline(
            config.model_name,
            config.random_state,
            tfidf_min_df=experiment["tfidf_min_df"],
            tfidf_max_features=experiment["tfidf_max_features"],
            tfidf_ngram_range=experiment["tfidf_ngram_range"],
            tfidf_sublinear_tf=experiment["tfidf_sublinear_tf"],
        )

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append(
            {
                "experiment": experiment["experiment"],
                "model": config.model_name,
                "tfidf_min_df": experiment["tfidf_min_df"],
                "tfidf_max_features": experiment["tfidf_max_features"],
                "tfidf_ngram_range": str(experiment["tfidf_ngram_range"]),
                "tfidf_sublinear_tf": experiment["tfidf_sublinear_tf"],
                "cv_accuracy_mean": cv_scores.mean(),
                "cv_accuracy_std": cv_scores.std(),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_macro_f1": f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
                "test_weighted_f1": f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
            }
        )

    experiments = pd.DataFrame(rows).sort_values(
        by="test_macro_f1",
        ascending=False,
    )
    experiments.to_csv(config.tfidf_experiments_path, index=False, encoding="utf-8")

    print(f"Zapisano eksperymenty TF-IDF do: {config.tfidf_experiments_path}")
    print("\nEksperymenty TF-IDF:")
    print(experiments)

    return config.tfidf_experiments_path


def save_evaluation_summary(
    df,
    metrics,
    errors_count,
    model_comparison_path,
    error_summary_path,
    config: TrainingConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model_comparison = pd.read_csv(model_comparison_path)
    error_summary = pd.read_csv(error_summary_path)

    best_model = model_comparison.iloc[0]
    main_model_name = MODEL_DISPLAY_NAMES.get(config.model_name, config.model_name)

    most_common_wrong_predictions = (
        error_summary["predicted_album"]
        .value_counts()
        .head(5)
        if not error_summary.empty
        else pd.Series(dtype=int)
    )

    lines = [
        "Lyrics Classifier - podsumowanie ewaluacji",
        "",
        f"Liczba albumów po filtrowaniu: {df['album'].nunique()}",
        f"Liczba utworów po filtrowaniu: {len(df)}",
        f"Minimalna liczba utworów na album: {config.min_songs_per_album}",
        f"TF-IDF min_df: {config.tfidf_min_df}",
        f"TF-IDF max_features: {config.tfidf_max_features}",
        f"TF-IDF ngram_range: {config.tfidf_ngram_range}",
        f"TF-IDF sublinear_tf: {config.tfidf_sublinear_tf}",
        "",
        f"Główny model: {main_model_name}",
        f"CV accuracy mean: {metrics['cv_accuracy_mean']:.4f}",
        f"CV accuracy std: {metrics['cv_accuracy_std']:.4f}",
        f"Test accuracy: {metrics['test_accuracy']:.4f}",
        f"Test macro F1: {metrics['test_macro_f1']:.4f}",
        f"Test weighted F1: {metrics['test_weighted_f1']:.4f}",
        "",
        "Najlepszy model w porównaniu:",
        f"{best_model['model']} "
        f"(test macro F1: {best_model['test_macro_f1']:.4f}, "
        f"test accuracy: {best_model['test_accuracy']:.4f})",
        "",
        f"Liczba błędnych predykcji w zbiorze testowym: {errors_count}",
        "",
        "Najczęściej przewidywane albumy wśród błędów:",
    ]

    if most_common_wrong_predictions.empty:
        lines.append("- brak błędów")
    else:
        for album, count in most_common_wrong_predictions.items():
            lines.append(f"- {album}: {count}")

    lines.extend(
        [
            "",
            "Interpretacja:",
            (
                "Spadek wyników względem wcześniejszej wersji 3-albumowej jest "
                "oczekiwany, ponieważ aktualne zadanie obejmuje wiele klas i "
                "mały, nierówny dataset."
            ),
            (
                "Model główny należy oceniać przez test macro F1, walidację "
                "krzyżową i analizę błędów, a nie samą accuracy."
            ),
            (
                "Błędy są rozproszone po wielu parach albumów, więc pojedynczy "
                "split testowy trzeba traktować ostrożnie."
            ),
        ]
    )

    config.evaluation_summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        "Zapisano podsumowanie ewaluacji do: "
        f"{config.evaluation_summary_path}"
    )