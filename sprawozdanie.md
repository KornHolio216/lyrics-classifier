# Sprawozdanie – Laboratorium 01

## 1. Temat ćwiczenia
Tworzenie modelu ML w Pythonie. Zapisywanie i wersjonowanie modelu.

## 2. Cel ćwiczenia
Celem ćwiczenia było stworzenie prostego modelu uczenia maszynowego w Pythonie, zapisanie go do pliku, przygotowanie skryptu do ponownego wczytywania modelu oraz omówienie podstaw wersjonowania i różnic pomiędzy środowiskiem deweloperskim i produkcyjnym. Zakres ten jest zgodny z treścią laboratorium: przygotowanie danych, trening modelu, zapis przez `joblib`, osobny skrypt do wczytania modelu oraz opis wersjonowania i środowiska produkcyjnego.

## 3. Link do repozytorium GitHub
Wstaw tutaj link do repozytorium po wrzuceniu projektu:

`https://github.com/twoj-login/lab01-taco-variant-b`

## 4. Opis problemu i zbioru danych
W projekcie wykorzystano zbiór danych zawierający teksty utworów Taco Hemingwaya. Dataset ma postać pliku CSV z kolumnami:
- `album`
- `title`
- `lyrics`

Pełny zbiór zawiera 165 utworów. Do realizacji projektu wybrano **wariant B**, czyli ograniczono problem do kilku klas, aby uzyskać prostszy i bardziej stabilny model klasyfikacyjny. Ostatecznie wykorzystano trzy albumy:
- `1-800-OŚWIECENIE`
- `Marmur`
- `LATARNIE WSZĘDZIE DAWNO ZGASŁY`

Taki wybór daje łącznie 57 utworów i pozwala zbudować model klasyfikacji tekstu przewidujący album na podstawie warstwy lirycznej utworu.

## 5. Analiza danych
Po wczytaniu danych program wyświetla:
- pierwsze 5 rekordów,
- rozmiar danych,
- typy kolumn,
- liczbę rekordów po odfiltrowaniu wybranych albumów,
- liczbę utworów w każdej klasie.

Rozkład klas po wyborze wariantu B:
- `1-800-OŚWIECENIE` – 26 utworów,
- `Marmur` – 16 utworów,
- `LATARNIE WSZĘDZIE DAWNO ZGASŁY` – 15 utworów.

## 6. Zastosowana metoda
Ponieważ problem dotyczy klasyfikacji tekstu, użyto biblioteki `scikit-learn` i pipeline'u składającego się z dwóch etapów:
1. `TfidfVectorizer` – zamiana tekstu na reprezentację numeryczną TF-IDF,
2. `LogisticRegression` – klasyfikator wieloklasowy.

Dane zostały podzielone na zbiory treningowy i testowy w proporcji 80% / 20% z użyciem `train_test_split`. Dodatkowo wykonano walidację krzyżową 5-fold.

## 7. Wyniki
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

Uzyskane wyniki wskazują, że model dobrze rozróżnia wybrane albumy na podstawie tekstu, choć ograniczona liczba próbek powoduje, że rezultat należy traktować jako demonstracyjny.

## 8. Zapisanie i wczytanie modelu
Wytrenowany model został zapisany do pliku:

`artifacts/model_v1.joblib`

Do projektu dodano osobny skrypt `load_model.py`, który:
- wczytuje model z pliku,
- wykonuje predykcję dla przykładowego tekstu,
- potwierdza poprawność serializacji modelu.

Taki sposób realizuje wymaganie laboratorium dotyczące zapisania modelu i jego późniejszego użycia.

## 9. Wersjonowanie modelu
Model został nazwany `model_v1.joblib`, co odpowiada pierwszej wersji artefaktu. W praktyce numer wersji należy zwiększać, gdy:
- zmienia się zbiór danych,
- zmienia się preprocessing,
- zmienia się architektura modelu lub hiperparametry,
- uzyskano istotną poprawę jakości,
- zmieniono zakres klas.

Repozytorium powinno zostać otagowane znacznikiem, np.:

```bash
git tag v1.0
git push origin v1.0
```

To odpowiada zaleceniom z treści zadania dotyczącym tagowania repozytorium i prowadzenia prostej polityki wersjonowania modelu.

## 10. Plik `.gitignore`
Do repozytorium dodano plik `.gitignore`, w którym pominięto m.in. folder środowiska wirtualnego i pliki tymczasowe. Jest to zgodne z zaleceniami z instrukcji laboratorium.

## 11. Różnice między środowiskiem deweloperskim a produkcyjnym
### Środowisko deweloperskie
- służy do eksperymentów,
- dane mogą być analizowane ręcznie,
- częściej dopuszcza się ręczne uruchamianie skryptów,
- zmiany w kodzie i zależnościach są częste.

### Środowisko produkcyjne
- musi być stabilne i powtarzalne,
- wymaga monitorowania jakości predykcji,
- trzeba kontrolować wersje bibliotek i modeli,
- ważne są automatyzacja wdrożeń, logowanie i bezpieczeństwo.

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
Ćwiczenie pozwoliło zrealizować pełny, prosty proces pracy z modelem ML: od wczytania danych, przez trening modelu klasyfikacyjnego, aż po zapis i ponowne wczytanie modelu. Zastosowanie rzeczywistego zbioru danych tekstowych uczyniło projekt bardziej praktycznym niż użycie gotowego zbioru demonstracyjnego. Jednocześnie niewielka liczba próbek pokazuje, że w rzeczywistych zastosowaniach konieczne byłoby rozszerzenie zbioru danych i dokładniejsze monitorowanie jakości modelu.
