from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

#config
DATA_PATH = Path("data/lyrics_data.csv")
MODEL_PATH = Path("artifacts/model_full_discography.joblib")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SONGS_PER_ALBUM = 4

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";;", engine="python")

    print("Rozmiar pełnego zbioru danych:", df.shape)
    print("\nTypy kolumn:")
    print(df.dtypes)

    required_columns = {"album", "title", "lyrics"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Brakujące kolumny w danych: {missing_columns}")

    df = df.dropna(subset=["album", "lyrics"]).copy()
    df["album"] = df["album"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["lyrics"] = df["lyrics"].astype(str).str.strip()
    df = df[df["lyrics"].str.len() > 0].copy()

    return df

def filter_albums(df: pd.DataFrame) -> pd.DataFrame:
    album_counts = df["album"].value_counts()

    print("\nLiczba utworów na album przed filtrowaniem:")
    print(album_counts)

    valid_albums = album_counts[album_counts >= MIN_SONGS_PER_ALBUM].index
    filtered = df[df["album"].isin(valid_albums)].copy()

    print(f"\nMinimalna liczba utworów na album: {MIN_SONGS_PER_ALBUM}")
    print(f"Liczba albumów po filtrowaniu: {filtered['album'].nunique()}")
    print(f"Liczba rekordów po filtrowaniu: {filtered.shape[0]}")

    print("\nAlbumy użyte do treningu:")
    for album in sorted(filtered["album"].unique()):
        print(f"- {album}")

    print("\nLiczba utworów na album po filtrowaniu:")
    print(filtered["album"].value_counts())

    if filtered["album"].nunique() < 2:
        raise ValueError("Po filtrowaniu zostało mniej niż 2 albumy. Zmniejsz MIN_SONGS_PER_ALBUM.")

    return filtered

def build_pipeline(classifier=None) -> Pipeline:
    if classifier is None:
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_df=0.95,
                    min_df=2,
                    max_features=5000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

#raporty
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
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "n_albums": int(n_albums),
        "n_songs": int(n_songs),
        "min_songs_per_album": int(MIN_SONGS_PER_ALBUM),
        "test_size": float(TEST_SIZE),
        "random_state": int(RANDOM_STATE),
        "cv_n_splits": int(n_splits),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_accuracy_scores": [float(score) for score in cv_scores],
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
    }

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    report = classification_report(y_test, predictions, zero_division=0)
    with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\nZapisano metryki do: {REPORTS_DIR / 'metrics.json'}")
    print(f"Zapisano raport klasyfikacji do: {REPORTS_DIR / 'classification_report.txt'}")

def save_confusion_matrix(y_test, predictions, labels) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 14))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    display.plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=False)

    plt.title("Confusion matrix - Lyrics Classifier")
    plt.tight_layout()

    output_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Zapisano confusion matrix do: {output_path}")

def save_errors_csv(X_test, y_test, predictions, pipeline) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    errors = pd.DataFrame({
        "lyrics": X_test,
        "true_album": y_test,
        "predicted_album": predictions,
    })

    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        confidence = probabilities.max(axis=1)
        errors["confidence"] = confidence
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

    output_path = REPORTS_DIR / "errors.csv"
    errors.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Zapisano błędne predykcje do: {output_path}")
    print(f"Liczba błędów w zbiorze testowym: {len(errors)}")

def save_error_summary_csv(y_test, predictions) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

    output_path = REPORTS_DIR / "error_summary.csv"
    error_summary.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Zapisano podsumowanie błędów do: {output_path}")

    return output_path

def save_model_comparison(X, y, X_train, X_test, y_train, y_test, cv) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "multinomial_nb": MultinomialNB(),
        "linear_svc": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    rows = []

    for model_name, classifier in models.items():
        pipeline = build_pipeline(classifier)

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

    output_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Zapisano porównanie modeli do: {output_path}")
    print("\nPorównanie modeli:")
    print(comparison)

    return output_path

def save_evaluation_summary(
    df,
    metrics,
    errors_count,
    model_comparison_path,
    error_summary_path,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_comparison = pd.read_csv(model_comparison_path)
    error_summary = pd.read_csv(error_summary_path)

    best_model = model_comparison.iloc[0]

    most_common_wrong_predictions = (
        error_summary["predicted_album"]
        .value_counts()
        .head(5)
    )

    lines = [
        "Lyrics Classifier - podsumowanie ewaluacji",
        "",
        f"Liczba albumów po filtrowaniu: {df['album'].nunique()}",
        f"Liczba utworów po filtrowaniu: {len(df)}",
        f"Minimalna liczba utworów na album: {MIN_SONGS_PER_ALBUM}",
        "",
        "Główny model: Logistic Regression",
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

    for album, count in most_common_wrong_predictions.items():
        lines.append(f"- {album}: {count}")

    lines.extend(
        [
            "",
            "Interpretacja:",
            (
                "Spadek wyników względem wcześniejszej wersji 3-albumowej jest oczekiwany, "
                "ponieważ aktualne zadanie obejmuje 15 klas i mały, nierówny dataset."
            ),
            (
                "Logistic Regression pozostaje najlepszym modelem referencyjnym, "
                "bo osiąga najwyższe test macro F1 w porównaniu modeli."
            ),
            (
                "Błędy są rozproszone po wielu parach albumów, ale Marmur pojawia się "
                "najczęściej jako błędna predykcja."
            ),
        ]
    )

    output_path = REPORTS_DIR / "evaluation_summary.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Zapisano podsumowanie ewaluacji do: {output_path}")

def main() -> None:
    df = load_data()
    filtered = filter_albums(df)

    X = filtered["lyrics"].astype(str)
    y = filtered["album"].astype(str)

    min_class_count = y.value_counts().min()
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise ValueError("Za mało przykładów w najmniejszej klasie do cross-validation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
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

    save_reports(
        y_test=y_test,
        predictions=predictions,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        cv_scores=cv_scores,
        n_albums=filtered["album"].nunique(),
        n_songs=filtered.shape[0],
        n_splits=n_splits,
    )

    save_confusion_matrix(
        y_test=y_test,
        predictions=predictions,
        labels=labels,
    )

    save_errors_csv(
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        pipeline=pipeline,
    )

    error_summary_path = save_error_summary_csv(
        y_test=y_test,
        predictions=predictions,
    )

    model_comparison_path = save_model_comparison(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=cv,
    )

    metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
    }

    save_evaluation_summary(
        df=filtered,
        metrics=metrics,
        errors_count=int((y_test != predictions).sum()),
        model_comparison_path=model_comparison_path,
        error_summary_path=error_summary_path,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel zapisano do: {MODEL_PATH}")

if __name__ == "__main__":
    main()