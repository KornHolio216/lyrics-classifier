from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/lyrics_data.csv")
MODEL_PATH = Path("artifacts/model_v1.joblib")
SELECTED_ALBUMS = [
    "1-800-OŚWIECENIE",
    "Marmur",
    "LATARNIE WSZĘDZIE DAWNO ZGASŁY",
]
RANDOM_STATE = 42

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";;", engine="python")
    print("Rozmiar pełnego zbioru danych:", df.shape)
    print("\nTypy kolumn:")
    print(df.dtypes)
    return df

def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_df=0.95,
                    max_features=3000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

def main() -> None:
    df = load_data()

    filtered = df[df["album"].isin(SELECTED_ALBUMS)].copy()

    print("\nWybrane albumy:")
    for album in SELECTED_ALBUMS:
        print(f"- {album}")

    print(f"\nLiczba rekordów po filtrowaniu do 3 albumów: {filtered.shape[0]}")

    print("\nLiczba utworów na album:")
    print(filtered["album"].value_counts())

    print("\nPierwsze 5 rekordów po filtrowaniu:")
    print(filtered[["album", "title"]].head())

    X = filtered["lyrics"].astype(str)
    y = filtered["album"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1)

    print("\nWyniki cross-validation (accuracy):", cv_scores)
    print("Średnia accuracy CV: {:.4f}".format(cv_scores.mean()))
    print("Odchylenie standardowe CV: {:.4f}".format(cv_scores.std()))

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy na zbiorze testowym: {:.4f}".format(accuracy))
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel zapisano do: {MODEL_PATH}")

if __name__ == "__main__":
    main()