import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


CSV_PATH = "data/processed/keypoint.csv"


def main():

    # =====================================
    # Load Dataset
    # =====================================

    dataset = pd.read_csv(CSV_PATH, header=None)

    print("=" * 60)
    print("First 5 Rows")
    print("=" * 60)
    print(dataset.head())

    # =====================================
    # Dataset Shape
    # =====================================

    print("\n" + "=" * 60)
    print("Dataset Shape")
    print("=" * 60)
    print(dataset.shape)

    # =====================================
    # Dataset Information
    # =====================================

    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)
    dataset.info()

    # =====================================
    # Missing Values
    # =====================================

    print("\n" + "=" * 60)
    print("Missing Values")
    print("=" * 60)
    print(dataset.isnull().sum())

    # =====================================
    # Separate Features and Labels
    # =====================================

    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]

    print("\n" + "=" * 60)
    print("Features (X)")
    print("=" * 60)
    print(X.head())

    print("\n" + "=" * 60)
    print("Labels (y)")
    print("=" * 60)
    print(y.head())

    # =====================================
    # Label Encoding
    # =====================================

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print("\n" + "=" * 60)
    print("Encoded Labels")
    print("=" * 60)
    print(y_encoded[:10])

    print("\n" + "=" * 60)
    print("Label Mapping")
    print("=" * 60)

    for index, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {index}")

    # =====================================
    # Train-Test Split
    # =====================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
    )

    print("\n" + "=" * 60)
    print("Training Set Shape")
    print("=" * 60)
    print(X_train.shape)

    print("\n" + "=" * 60)
    print("Testing Set Shape")
    print("=" * 60)
    print(X_test.shape)

    print("\n" + "=" * 60)
    print("Training Labels Shape")
    print("=" * 60)
    print(y_train.shape)

    print("\n" + "=" * 60)
    print("Testing Labels Shape")
    print("=" * 60)
    print(y_test.shape)


if __name__ == "__main__":
    main()