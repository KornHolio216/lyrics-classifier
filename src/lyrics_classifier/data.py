import pandas as pd

from .config import TrainingConfig


def load_data(config: TrainingConfig) -> pd.DataFrame:
    df = pd.read_csv(config.data_path, sep=";;", engine="python")

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


def filter_albums(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    album_counts = df["album"].value_counts()

    print("\nLiczba utworów na album przed filtrowaniem:")
    print(album_counts)

    valid_albums = album_counts[
        album_counts >= config.min_songs_per_album
    ].index
    filtered = df[df["album"].isin(valid_albums)].copy()

    print(f"\nMinimalna liczba utworów na album: {config.min_songs_per_album}")
    print(f"Liczba albumów po filtrowaniu: {filtered['album'].nunique()}")
    print(f"Liczba rekordów po filtrowaniu: {filtered.shape[0]}")

    print("\nAlbumy użyte do treningu:")
    for album in sorted(filtered["album"].unique()):
        print(f"- {album}")

    print("\nLiczba utworów na album po filtrowaniu:")
    print(filtered["album"].value_counts())

    if filtered["album"].nunique() < 2:
        raise ValueError(
            "Po filtrowaniu zostało mniej niż 2 albumy. "
            "Zmniejsz minimalną liczbę utworów."
        )

    return filtered