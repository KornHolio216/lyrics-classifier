from pathlib import Path
import joblib

MODEL_PATH = Path("artifacts/model_v1.joblib")


def main() -> None:
    model = joblib.load(MODEL_PATH)

    sample_text = """
    Warszawa nocą, szybkie tempo życia, kluby, taxi, miasto i znajomi.
    Dużo obserwacji o codzienności, relacjach i presji dużego miasta.
    """

    prediction = model.predict([sample_text])[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([sample_text])[0]
        classes = model.classes_

    print("Model został poprawnie wczytany z pliku.")
    print("\nPrzykładowy tekst:")
    print(sample_text.strip())
    print("\nPrzewidziany album:", prediction)

    if probabilities is not None:
        print("\nPrawdopodobieństwa klas:")
        for label, prob in sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True):
            print(f"- {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
