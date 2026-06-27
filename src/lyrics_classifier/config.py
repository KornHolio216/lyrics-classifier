from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    data_path: Path = Path("data/lyrics_data.csv")
    model_path: Path = Path("artifacts/model_full_discography.joblib")
    output_dir: Path = Path("reports")
    random_state: int = 42
    test_size: float = 0.2
    min_songs_per_album: int = 4
    model_name: str = "logistic_regression"
    max_cv_splits: int = 5
    tfidf_min_df: int = 2
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_sublinear_tf: bool = True

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"

    @property
    def classification_report_path(self) -> Path:
        return self.output_dir / "classification_report.txt"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.output_dir / "confusion_matrix.png"

    @property
    def errors_path(self) -> Path:
        return self.output_dir / "errors.csv"

    @property
    def error_summary_path(self) -> Path:
        return self.output_dir / "error_summary.csv"

    @property
    def model_comparison_path(self) -> Path:
        return self.output_dir / "model_comparison.csv"

    @property
    def evaluation_summary_path(self) -> Path:
        return self.output_dir / "evaluation_summary.txt"

    @property
    def top_features_path(self) -> Path:
        return self.output_dir / "top_features_by_album.csv"

    @property
    def tfidf_experiments_path(self) -> Path:
        return self.output_dir / "tfidf_experiments.csv"