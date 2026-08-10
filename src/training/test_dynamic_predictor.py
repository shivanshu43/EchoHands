import numpy as np

from src.core.dynamic_predictor import DynamicPredictor


def main():

    predictor = DynamicPredictor()

    print("\n========== Dynamic Predictor Test ==========\n")

    # ------------------------------------------
    # Load one saved J sequence
    # ------------------------------------------

    j_path = (
        "data/processed/"
        "dynamic_sequences/"
        "J/LEFT/J_LEFT_001.npz"
    )

    j_data = np.load(
        j_path,
        allow_pickle=True
    )

    j_sequence = j_data["sequence"]

    print(
        "J sequence shape:",
        j_sequence.shape
    )

    predicted_class, confidence = (
        predictor.predict(
            j_sequence
        )
    )

    print(
        f"J Prediction: "
        f"{predicted_class}"
    )

    print(
        f"J Confidence: "
        f"{confidence * 100:.2f}%"
    )

    # ------------------------------------------
    # Load one saved Z sequence
    # ------------------------------------------

    z_path = (
        "data/processed/"
        "dynamic_sequences/"
        "Z/LEFT/Z_LEFT_001.npz"
    )

    z_data = np.load(
        z_path,
        allow_pickle=True
    )

    z_sequence = z_data["sequence"]

    print(
        "\nZ sequence shape:",
        z_sequence.shape
    )

    predicted_class, confidence = (
        predictor.predict(
            z_sequence
        )
    )

    print(
        f"Z Prediction: "
        f"{predicted_class}"
    )

    print(
        f"Z Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        "\n============================================\n"
    )


if __name__ == "__main__":
    main()