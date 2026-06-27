from pathlib import Path
import joblib

MODEL_PATH = Path("artifacts/model_full_discography.joblib")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH}. "
            "Uruchom najpierw `python train_model.py`."
        )

    model = joblib.load(MODEL_PATH)

    sample_text = """
    Głucha noc, na ulicach ciągle cicho.
    Noc się toczy, tak jak każda inna.
    Znowu pokłóciłem się z taryfą.
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
        for label, prob in sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"- {label}: {prob:.4f}")


if __name__ == "__main__":
    main()