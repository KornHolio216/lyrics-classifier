import pytest
from sklearn.pipeline import Pipeline

from lyrics_classifier.modeling import build_pipeline


def test_build_pipeline_uses_tfidf_and_classifier_steps():
    pipeline = build_pipeline("logistic_regression", random_state=42)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="Nieznany model"):
        build_pipeline("unknown_model", random_state=42)