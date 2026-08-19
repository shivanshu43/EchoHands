import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


CSV_PATH = "data/processed/keypoints_geometric.csv"


def load_dataset():

    # Load CSV
    dataframe = pd.read_csv(
        CSV_PATH,
        header=None,
    )

    # First column = labels
    labels = dataframe.iloc[:, 0]

    # Remaining columns = features
    features = dataframe.iloc[:, 1:]

    # Encode labels
    encoder = LabelEncoder()

    encoded_labels = encoder.fit_transform(labels)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        encoded_labels,
        test_size=0.20,
        random_state=42,
        stratify=encoded_labels,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
    )