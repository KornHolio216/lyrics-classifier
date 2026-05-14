# Lyrics Classifier - klasyfikacja albumów Taco Hemingwaya

Projekt ML/NLP służący do klasyfikacji albumu Taco Hemingwaya na podstawie tekstu piosenki.

Projekt powstał jako laboratorium z tworzenia, zapisywania i wersjonowania modelu ML w Pythonie, a następnie został rozszerzony w kierunku bardziej kompletnego projektu portfolio.

## Problem

Na podstawie tekstu piosenki model przewiduje, z którego albumu pochodzi utwór.

Pierwsza wersja projektu działała na ręcznie wybranych 3 albumach:

- `1-800-OŚWIECENIE`
- `Marmur`
- `LATARNIE WSZĘDZIE DAWNO ZGASŁY`

Aktualna wersja rozszerza trening na całą dyskografię dostępną w zbiorze danych. Model automatycznie wybiera albumy, które mają co najmniej 4 utwory w dataspecie.

## Dane

Projekt korzysta z pliku:

```text
data/lyrics_data.csv
```

Dataset zawiera 165 rekordów i 3 kolumny:

```text
album
title
lyrics
```

Po usunięciu rekordów niespełniających warunków oraz zastosowaniu progu `MIN_SONGS_PER_ALBUM = 4`, do treningu wykorzystywane są:

```text
15 albumów
164 utwory
```

Albumy użyte w aktualnej wersji:

- `0,25 mg EP`
- `1-800-OŚWIECENIE`
- `Café Belga`
- `Europa`
- `Flagey EP`
- `Jarmark`
- `LATARNIE WSZĘDZIE DAWNO ZGASŁY`
- `Marmur`
- `POCZTÓWKA Z WWA, LATO '19`
- `SOMA 0,5 mg`
- `Szprycer`
- `Trójkąt Warszawski`
- `Umowa o dzieło`
- `WOSK EP`
- `Young Hems`

## Aktualny pipeline

Model jest zbudowany jako pipeline `scikit-learn`:

```text
TfidfVectorizer -> LogisticRegression
```

Wykorzystywane elementy:

- czyszczenie podstawowych braków w danych,
- automatyczne filtrowanie albumów według minimalnej liczby utworów,
- reprezentacja tekstu przez TF-IDF,
- unigramy i bigramy,
- regresja logistyczna z `class_weight="balanced"`,
- walidacja krzyżowa `StratifiedKFold`,
- zapis modelu do pliku `.joblib`,
- zapis metryk do katalogu `reports/`.

## Wyniki aktualnej wersji

Konfiguracja:

```text
MIN_SONGS_PER_ALBUM = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV folds = 4
```

Wyniki:

```text
CV accuracy mean: 0.6098
CV accuracy std: 0.0622
Test accuracy: 0.5455
Test macro F1: 0.4583
Test weighted F1: 0.4885
```

Wyniki są zapisywane do:

```text
reports/metrics.json
reports/classification_report.txt
```

Aktualny model jest zapisywany jako:

```text
artifacts/model_full_discography.joblib
```

## Uruchomienie

### 1. Utworzenie środowiska

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instalacja zależności

```bash
pip install -r requirements.txt
```

### 3. Trening modelu

```bash
python train_model.py
```

Po uruchomieniu skrypt:

- wczytuje dane,
- pokazuje rozkład utworów po albumach,
- filtruje albumy z mniej niż 4 utworami,
- trenuje model,
- wyświetla metryki,
- zapisuje raporty,
- zapisuje model.

### 4. Testowe wczytanie modelu

```bash
python load_model.py
```

## Struktura projektu

```text
.
├── data/
│   └── lyrics_data.csv
├── artifacts/
│   └── model_full_discography.joblib
├── reports/
│   ├── metrics.json
│   └── classification_report.txt
├── train_model.py
├── load_model.py
├── requirements.txt
├── README.md
└── sprawozdanie.md
```

## Wersjonowanie modelu

Pierwsza wersja modelu:

```text
artifacts/model_v1.joblib
```

Aktualna wersja modelu dla rozszerzonego datasetu:

```text
artifacts/model_full_discography.joblib
```

Kolejne wersje modelu powinny być tworzone po zmianie:

- danych,
- preprocessing'u,
- parametrów TF-IDF,
- algorytmu klasyfikacji,
- sposobu ewaluacji,
- minimalnej liczby utworów na album.

Przykład tagu Git:

```bash
git tag v2.0
git push origin v2.0
```

## Ograniczenia

Dataset jest niewielki, a część albumów ma bardzo mało przykładów. Przykładowo `Flagey EP` ma 4 utwory w całym zbiorze, więc po podziale train/test model ma bardzo mało danych do nauczenia się tej klasy.

Z tego powodu pojedynczy wynik na zbiorze testowym należy traktować ostrożnie. Ważniejsze są:

- walidacja krzyżowa,
- macro F1,
- analiza błędów,
- confusion matrix,
- porównanie kilku modeli.