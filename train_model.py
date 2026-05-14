from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

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

def build_pipeline() -> Pipeline:
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
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
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

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel zapisano do: {MODEL_PATH}")

if __name__ == "__main__":
    main()