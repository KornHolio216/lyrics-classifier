# LAB01 - klasyfikacja albumów Taco Hemingwaya

Projekt realizuje laboratorium z tworzenia, zapisywania i wersjonowania modelu ML w Pythonie.

## Problem
Na podstawie tekstu utworu model przewiduje, z którego albumu pochodzi piosenka.

Wybrane albumy (wariant B):
- 1-800-OŚWIECENIE
- Marmur
- LATARNIE WSZĘDZIE DAWNO ZGASŁY

## Uruchomienie
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
python train_model.py
python load_model.py
```

## Struktura projektu
- `train_model.py` - trening i zapis modelu
- `load_model.py` - wczytanie modelu i predykcja testowa
- `data/lyrics_data.csv` - dane wejściowe
- `artifacts/model_v1.joblib` - zapisany model
- `sprawozdanie.md` - gotowe sprawozdanie w Markdown

## Wersjonowanie modelu
- `model_v1.joblib` - pierwsza wersja modelu
- kolejne wersje należy tworzyć po zmianie danych, modelu, hiperparametrów lub preprocessing'u

Przykład tagu Git:
```bash
git tag v1.0
git push origin v1.0
```
