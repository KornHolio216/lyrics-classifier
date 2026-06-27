from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


MODEL_NAMES = (
    "logistic_regression",
    "linear_svc",
    "multinomial_nb",
    "dummy_most_frequent",
)


MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "linear_svc": "Linear SVC",
    "multinomial_nb": "Multinomial NB",
    "dummy_most_frequent": "Dummy Most Frequent",
}


def build_classifier(model_name: str, random_state: int):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )

    if model_name == "linear_svc":
        return LinearSVC(
            class_weight="balanced",
            random_state=random_state,
        )

    if model_name == "multinomial_nb":
        return MultinomialNB()

    if model_name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")

    raise ValueError(
        f"Nieznany model: {model_name}. "
        f"Dostępne modele: {', '.join(MODEL_NAMES)}"
    )


def build_pipeline(
    model_name: str,
    random_state: int,
    *,
    tfidf_min_df: int = 2,
    tfidf_max_features: int | None = 5000,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
    tfidf_sublinear_tf: bool = True,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_df=0.95,
                    min_df=tfidf_min_df,
                    max_features=tfidf_max_features,
                    ngram_range=tfidf_ngram_range,
                    sublinear_tf=tfidf_sublinear_tf,
                ),
            ),
            (
                "classifier",
                build_classifier(model_name, random_state),
            ),
        ]
    )