from argparse import Namespace
from pathlib import Path

from lyrics_classifier.cli import build_config


def test_build_config_maps_cli_arguments_to_training_config():
    args = Namespace(
        data_path=Path("custom.csv"),
        model_path=Path("model.joblib"),
        output_dir=Path("custom_reports"),
        random_state=123,
        test_size=0.25,
        min_songs=5,
        model="linear_svc",
        max_cv_splits=3,
        tfidf_min_df=1,
        tfidf_max_features=3000,
        tfidf_ngram_max=1,
        no_sublinear_tf=True,
    )

    config = build_config(args)

    assert config.data_path == Path("custom.csv")
    assert config.output_dir == Path("custom_reports")
    assert config.min_songs_per_album == 5
    assert config.model_name == "linear_svc"
    assert config.tfidf_ngram_range == (1, 1)
    assert config.tfidf_sublinear_tf is False