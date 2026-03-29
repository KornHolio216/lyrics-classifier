# Nowoczesne Technologie Przetwarzania Danych – Laboratorium 01

## Prowadzący: mgr inż. Leonid Rusanov
## Student: Mateusz Machowski, grupa 4
## 1.Temat ćwiczenia
Tworzenie modelu ML w Pythonie. Zapisywanie i wersjonowanie modelu.

## 2.Cel ćwiczenia
Celem ćwiczenia było stworzenie prostego modelu uczenia maszynowego w Pythonie, zapisanie go do pliku, przygotowanie skryptu do ponownego wczytywania modelu oraz omówienie podstaw wersjonowania i różnic pomiędzy środowiskiem deweloperskim i produkcyjnym.

## 3.Link do repozytorium GitHub
`https://github.com/KornHolio216/lyrics-classifier`

## 4.Opis problemu i zbioru danych
W projekcie wykorzystałem zbiór danych zawierający teksty utworów Taco Hemingwaya. Dataset ma postać pliku CSV z kolumnami:
- `album`
- `title`
- `lyrics`

Pełny zbiór zawiera 165 utworów. Do realizacji projektu ograniczyłem problem do kilku klas, aby uzyskać prostszy i bardziej stabilny model. Ostatecznie wykorzystałem trzy albumy:
- `1-800-OŚWIECENIE`
- `Marmur`
- `LATARNIE WSZĘDZIE DAWNO ZGASŁY`

Taki wybór daje łącznie 57 utworów i pozwala zbudować model klasyfikacji tekstu który przewiduje album na podstawie tekstów.

## 5.Analiza danych
Po wczytaniu danych wyświetlam:
- pierwsze 5 rekordów,
- rozmiar danych,
- typy kolumn,
- liczbę rekordów po odfiltrowaniu wybranych albumów,
- liczbę utworów w każdej klasie.

Rozkład klas:
- `1-800-OŚWIECENIE` – 26 utworów,
- `Marmur` – 16 utworów,
- `LATARNIE WSZĘDZIE DAWNO ZGASŁY` – 15 utworów.

## 6.Zastosowana metoda
Ponieważ problem dotyczy klasyfikacji tekstu, użyłem biblioteki `scikit-learn` i pipeline'u składającego się z dwóch etapów:
1. `TfidfVectorizer` – zamiana tekstu na reprezentację numeryczną TF-IDF,
2. `LogisticRegression` – klasyfikator wieloklasowy.

Dane podzieliłem na zbiór treningowy i testowy w proporcji 80% / 20% z użyciem `train_test_split`. Dodatkowo wykonałem walidację krzyżową 5-fold.

## 7.Wyniki
Dla przygotowanego modelu uzyskano następujące wyniki:
- średnia accuracy z 5-fold cross-validation: **0.7894**,
- odchylenie standardowe accuracy: **0.1033**,
- accuracy na zbiorze testowym: **0.9167**.

Raport klasyfikacji dla zbioru testowego:

```text
                                precision    recall  f1-score   support

              1-800-OŚWIECENIE       1.00      0.83      0.91         6
LATARNIE WSZĘDZIE DAWNO ZGASŁY       0.75      1.00      0.86         3
                        Marmur       1.00      1.00      1.00         3

                      accuracy                           0.92        12
                     macro avg       0.92      0.94      0.92        12
                  weighted avg       0.94      0.92      0.92        12
```
Wyniki pokazują, że model dobrze rozróżnia albumy na podstawie tekstu, choć ograniczona liczba próbek powoduje, że rezultat należy traktować jako demonstracyjny.

## 8.Zapisanie i wczytanie modelu
Wytrenowany model został zapisany do pliku:

`artifacts/model_v1.joblib`

Do projektu dodałem osobny skrypt `load_model.py`, który:
- wczytuje model z pliku,
- wykonuje predykcję dla przykładowego tekstu,
- potwierdza poprawność serializacji modelu.

Podczas testu model został poprawnie załadowany z pliku i użyty do predykcji dla przykładowego fragmentu tekstu z utworu należącego do jednej z analizowanych klas. Dla użytego przykładu model przewidział album:

- `1-800-OŚWIECENIE`

Uzyskane prawdopodobieństwa klas dla przykładowego tekstu:
- `1-800-OŚWIECENIE` – **0.3833**
- `LATARNIE WSZĘDZIE DAWNO ZGASŁY` – **0.3119**
- `Marmur` – **0.3048**

## 9.Wersjonowanie modelu
Model nazwałem `model_v1.joblib`, co odpowiada pierwszej wersji. W praktyce numer wersji zwiększamy, gdy:
- zmienia się zbiór danych,
- zmienia się preprocessing,
- zmienia się architektura modelu lub parametry,
- uzyskano istotną poprawę jakości,
- zmieniono zakres klas.

Repozytorium tagujemy znacznikiem, np:

```bash
git tag v1.0
git push origin v1.0
```

## 10.Plik `.gitignore`
Do repozytorium dodałem plik `.gitignore`, w którym pomijam m.in. folder środowiska wirtualnego i pliki tymczasowe.

## 11.Różnice między środowiskiem deweloperskim a produkcyjnym
### Środowisko deweloperskie
- służy do eksperymentów,
- dane mogą być analizowane ręcznie,
- częściej dopuszcza się ręczne uruchamianie skryptów,
- zmiany w kodzie i zależnościach są częste.

### Środowisko produkcyjne
- musi być stabilne i powtarzalne,
- wymaga monitorowania jakości predykcji,
- trzeba kontrolować wersje bibliotek i modeli,
- ważna jest automatyzacja wdrożeń, logowanie i bezpieczeństwo.

### Główne wyzwania
- **data drift** – dane w produkcji mogą różnić się od treningowych,
- **model drift** – jakość modelu może z czasem spadać,
- **zarządzanie zależnościami** – inna wersja biblioteki może wpływać na działanie modelu,
- **retraining** – model może wymagać ponownego uczenia,
- **wdrożenie i monitoring** – potrzebne są logi, metryki i alerty.

### Sposoby radzenia sobie z problemami
- stosowanie pliku `requirements.txt`,
- wersjonowanie modeli i danych,
- okresowe ponowne uczenie modelu,
- monitorowanie jakości predykcji,
- automatyzacja testów i wdrożeń.

## 12. Wnioski końcowe
Ćwiczenie pozwoliło mi zrealizować pełny, prosty proces pracy 
z modelem ML: od wczytania danych, przez trening modelu klasyfikacyjnego,
aż po zapis i ponowne wczytanie modelu. Zastosowanie rzeczywistego zbioru
danych tekstowych uczyniło projekt bardziej praktycznym niż użycie 
gotowego zbioru demonstracyjnego. Jednocześnie niewielka liczba próbek
pokazuje, że w rzeczywistych zastosowaniach konieczne byłoby rozszerzenie
zbioru danych i dokładniejsze monitorowanie jakości modelu.
