import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import TrainingConfig
from .data import filter_albums, load_data
from .modeling import build_pipeline
from .reports import (
    save_confusion_matrix,
    save_error_summary_csv,
    save_errors_csv,
    save_evaluation_summary,
    save_model_comparison,
    save_reports,
    save_tfidf_experiments,
    save_top_features_csv,
)


def run_training(config: TrainingConfig) -> dict:
    df = load_data(config)
    filtered = filter_albums(df, config)

    X = filtered["lyrics"].astype(str)
    y = filtered["album"].astype(str)

    min_class_count = y.value_counts().min()
    n_splits = min(config.max_cv_splits, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "Za mało przykładów w najmniejszej klasie do cross-validation."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(
        config.model_name,
        config.random_state,
        tfidf_min_df=config.tfidf_min_df,
        tfidf_max_features=config.tfidf_max_features,
        tfidf_ngram_range=config.tfidf_ngram_range,
        tfidf_sublinear_tf=config.tfidf_sublinear_tf,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    print("\nWyniki cross-validation (accuracy):", cv_scores)
    print("Średnia accuracy CV: {:.4f}".format(cv_scores.mean()))
    print("Odchylenie standardowe CV: {:.4f}".format(cv_scores.std()))

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    labels = sorted(y.unique())

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print("\nAccuracy na zbiorze testowym: {:.4f}".format(accuracy))
    print("Macro F1 na zbiorze testowym: {:.4f}".format(macro_f1))
    print("Weighted F1 na zbiorze testowym: {:.4f}".format(weighted_f1))

    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, predictions, zero_division=0))

    metrics = save_reports(
        y_test=y_test,
        predictions=predictions,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        cv_scores=cv_scores,
        n_albums=filtered["album"].nunique(),
        n_songs=filtered.shape[0],
        n_splits=n_splits,
        config=config,
    )

    save_confusion_matrix(
        y_test=y_test,
        predictions=predictions,
        labels=labels,
        config=config,
    )

    errors_count = save_errors_csv(
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        pipeline=pipeline,
        config=config,
    )

    error_summary_path = save_error_summary_csv(
        y_test=y_test,
        predictions=predictions,
        config=config,
    )

    model_comparison_path = save_model_comparison(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=cv,
        config=config,
    )

    save_top_features_csv(
        pipeline=pipeline,
        config=config,
    )

    save_tfidf_experiments(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=cv,
        config=config,
    )

    save_evaluation_summary(
        df=filtered,
        metrics=metrics,
        errors_count=errors_count,
        model_comparison_path=model_comparison_path,
        error_summary_path=error_summary_path,
        config=config,
    )

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.model_path)
    print(f"\nModel zapisano do: {config.model_path}")

    return metrics