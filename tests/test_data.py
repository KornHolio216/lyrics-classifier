import pandas as pd

from lyrics_classifier.config import TrainingConfig
from lyrics_classifier.data import filter_albums


def test_filter_albums_keeps_only_albums_with_enough_songs():
    df = pd.DataFrame(
        {
            "album": ["A", "A", "B", "C", "C"],
            "title": ["a1", "a2", "b1", "c1", "c2"],
            "lyrics": ["tekst"] * 5,
        }
    )
    config = TrainingConfig(min_songs_per_album=2)

    filtered = filter_albums(df, config)

    assert sorted(filtered["album"].unique()) == ["A", "C"]
    assert len(filtered) == 4