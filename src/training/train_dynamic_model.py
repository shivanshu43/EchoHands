import os

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder


DATA_DIR = "data/processed/dynamic/split"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "dynamic_lstm.keras"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "dynamic_label_encoder.npy"
)


def load_data():

    X_train = np.load(
        os.path.join(DATA_DIR, "X_train.npy")
    )

    y_train = np.load(
        os.path.join(DATA_DIR, "y_train.npy")
    )

    X_val = np.load(
        os.path.join(DATA_DIR, "X_val.npy")
    )

    y_val = np.load(
        os.path.join(DATA_DIR, "y_val.npy")
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val
    )


def build_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(40, 70)
        ),

        tf.keras.layers.LSTM(
            64
        ),

        tf.keras.layers.Dropout(
            0.3
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            2,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    print("\n========== Loading Data ==========\n")

    (
        X_train,
        y_train,
        X_val,
        y_val
    ) = load_data()

    print(
        "Training:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    # ==========================================
    # Encode labels
    # ==========================================

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(
        y_train
    )

    y_val_encoded = encoder.transform(
        y_val
    )

    print(
        "\nClasses:",
        encoder.classes_
    )

    # ==========================================
    # Build model
    # ==========================================

    model = build_model()

    print("\n========== Model ==========\n")

    model.summary()

    # ==========================================
    # Training
    # ==========================================

    print("\n========== Training ==========\n")

    history = model.fit(
        X_train,
        y_train_encoded,

        validation_data=(
            X_val,
            y_val_encoded
        ),

        epochs=40,
        batch_size=16,

        verbose=1
    )

    # ==========================================
    # Save model
    # ==========================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    np.save(
        ENCODER_PATH,
        encoder.classes_
    )

    print("\n========== Training Complete ==========\n")

    print(
        "Model saved:",
        MODEL_PATH
    )

    print(
        "Encoder saved:",
        ENCODER_PATH
    )


if __name__ == "__main__":
    main()