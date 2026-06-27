import argparse
from pathlib import Path

from .config import TrainingConfig
from .evaluation import run_training
from .modeling import MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trenuje i ewaluuje klasyfikator albumów na podstawie lyrics.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/lyrics_data.csv"),
        help="Ścieżka do pliku CSV z danymi.",
    )
    parser.add_argument(
        "--min-songs",
        type=int,
        default=4,
        help="Minimalna liczba utworów wymagana do użycia albumu.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Udział zbioru testowego, np. 0.2.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="logistic_regression",
        help="Model używany jako główny klasyfikator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Katalog wyjściowy dla raportów.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/model_full_discography.joblib"),
        help="Ścieżka zapisu modelu joblib.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed losowości używany w podziale danych i modelach.",
    )
    parser.add_argument(
        "--max-cv-splits",
        type=int,
        default=5,
        help="Maksymalna liczba foldów w StratifiedKFold.",
    )
    parser.add_argument(
        "--tfidf-min-df",
        type=int,
        default=2,
        help="Minimalna liczba dokumentów dla cechy TF-IDF.",
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=5000,
        help="Maksymalna liczba cech TF-IDF.",
    )
    parser.add_argument(
        "--tfidf-ngram-max",
        type=int,
        choices=(1, 2),
        default=2,
        help="Maksymalny rozmiar n-gramu TF-IDF.",
    )
    parser.add_argument(
        "--no-sublinear-tf",
        action="store_true",
        help="Wyłącza sublinear_tf w TfidfVectorizer.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        data_path=args.data_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        min_songs_per_album=args.min_songs,
        model_name=args.model,
        max_cv_splits=args.max_cv_splits,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_ngram_range=(1, args.tfidf_ngram_max),
        tfidf_sublinear_tf=not args.no_sublinear_tf,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run_training(config)