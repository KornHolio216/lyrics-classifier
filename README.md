# Lyrics Classifier - klasyfikacja albumów Taco Hemingwaya

Projekt ML/NLP służący do klasyfikacji albumu Taco Hemingwaya na 
podstawie tekstu piosenki.

## Problem

Na podstawie tekstu piosenki model przewiduje, z którego albumu 
Taco Hemingwaya pochodzi utwór.

Pierwsza wersja projektu działała na ręcznie wybranych 3 albumach:

- `1-800-OŚWIECENIE`
- `Marmur`
- `LATARNIE WSZĘDZIE DAWNO ZGASŁY`

Aktualna wersja rozszerza trening na pełniejszą dyskografię
dostępną w zbiorze danych. Zamiast ręcznie wskazywać albumy, 
skrypt automatycznie wybiera albumy, które mają co najmniej 4 utwory w zbiorze danych.
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
- usuwanie rekordów bez albumu lub bez tekstu,
- automatyczne filtrowanie albumów według minimalnej liczby utworów,
- reprezentacja tekstu przez TF-IDF,
- unigramy i bigramy,
- regresja logistyczna z `class_weight="balanced"`,
- podział train/test ze stratyfikacją,
- walidacja krzyżowa `StratifiedKFold`,
- zapis modelu do pliku `.joblib`,
- zapis metryk i raportów do katalogu `reports/`.

Aktualny model główny:

```text
LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
```

## Wyniki aktualnej wersji

Konfiguracja:

```text
MIN_SONGS_PER_ALBUM = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV folds = 4
```

Wyniki głównego modelu `Logistic Regression`:

```text
CV accuracy mean: 0.6098
CV accuracy std: 0.0622
Test accuracy: 0.5455
Test macro F1: 0.4583
Test weighted F1: 0.4885
```

Spadek wyników względem wcześniejszej wersji 3-albumowej jest oczekiwany. Aktualne zadanie jest trudniejsze, ponieważ model rozróżnia 15 klas, dataset jest niewielki, a część albumów ma mało przykładów.

Wyniki są zapisywane do:

```text
reports/metrics.json
reports/classification_report.txt
reports/evaluation_summary.txt
```

Aktualny model jest zapisywany lokalnie jako:

```text
artifacts/model_full_discography.joblib
```

Pliki `.joblib` nie są trzymane w repozytorium, ponieważ są artefaktami generowanymi lokalnie.

## Porównanie modeli

Projekt porównuje kilka klasycznych modeli tekstowych:

- `DummyClassifier` jako baseline,
- `LogisticRegression`,
- `MultinomialNB`,
- `LinearSVC`.

Wyniki są zapisywane do:

```text
reports/model_comparison.csv
```

Aktualne porównanie:

| Model | CV accuracy mean | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6098 | 0.5455 | 0.4583 | 0.4885 |
| Linear SVC | 0.6159 | 0.5455 | 0.4286 | 0.4784 |
| Multinomial NB | 0.2073 | 0.1818 | 0.0847 | 0.0713 |
| Dummy Most Frequent | 0.1585 | 0.1515 | 0.0175 | 0.0399 |

Najlepszym modelem referencyjnym pozostaje `Logistic Regression`. `LinearSVC` osiąga takie samo `test accuracy`, ale niższe `test macro F1`, dlatego nie zastępuje aktualnego modelu głównego.

Baseline `DummyClassifier` pokazuje, że proste przewidywanie najczęstszej klasy daje około `0.1515` accuracy, więc modele tekstowe rzeczywiście uczą się sygnału z danych.

## Analiza błędów

Skrypt zapisuje dodatkowe raporty pomagające zrozumieć pomyłki modelu:

```text
reports/confusion_matrix.png
reports/errors.csv
reports/error_summary.csv
```

`errors.csv` zawiera konkretne błędne predykcje wraz z prawdziwym albumem, przewidzianym albumem, confidence oraz podglądem tekstu.

`error_summary.csv` agreguje błędy po parach:

```text
true_album -> predicted_album
```

W aktualnym podziale testowym model popełnia 15 błędów. Najczęściej przewidywane albumy wśród błędów:

```text
Marmur: 5
Café Belga: 3
POCZTÓWKA Z WWA, LATO '19: 2
SOMA 0,5 mg: 1
LATARNIE WSZĘDZIE DAWNO ZGASŁY: 1
```

Najważniejszy wniosek z analizy błędów: model zbyt często przewiduje `Marmur` dla utworów z innych albumów. Błędy są jednak rozproszone po wielu parach albumów, więc problem nie sprowadza się do jednej dominującej pomyłki.

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

### 3. Trening modelu i wygenerowanie raportów

```bash
python train_model.py
```

Po uruchomieniu skrypt:

- wczytuje dane,
- pokazuje rozkład utworów po albumach,
- filtruje albumy z mniej niż 4 utworami,
- dzieli dane na train/test,
- trenuje główny model,
- liczy metryki,
- generuje confusion matrix,
- zapisuje błędne predykcje,
- tworzy podsumowanie błędów,
- porównuje kilka modeli,
- zapisuje tekstowe podsumowanie ewaluacji,
- zapisuje model lokalnie do `artifacts/`.

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
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── errors.csv
│   ├── error_summary.csv
│   ├── model_comparison.csv
│   └── evaluation_summary.txt
├── train_model.py
├── load_model.py
├── requirements.txt
├── README.md
└── sprawozdanie.md
```

Uwaga: katalog `artifacts/` oraz pliki `.joblib` mogą nie być widoczne w repozytorium po sklonowaniu projektu, ponieważ model jest generowany lokalnie i ignorowany przez Git.

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
- preprocessingu,
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

Wynik `0.5455` accuracy nie powinien być interpretowany w oderwaniu od kontekstu. Zadanie obejmuje 15 klas, a dane są małe i nierówne. W projekcie ważniejsze jest pokazanie pełnego procesu ML: przygotowania danych, uczciwej ewaluacji, analizy błędów i świadomego wyboru modelu.