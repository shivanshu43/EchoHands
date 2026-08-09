import numpy as np


FILE_PATH = (
    "data/processed/dynamic_sequences/"
    "J/RIGHT/J_RIGHT_001.npz"
)


def main():

    data = np.load(FILE_PATH, allow_pickle=True)

    sequence = data["sequence"]

    label = data["label"]
    hand = data["hand"]

    print("\n========== Sequence Information ==========\n")

    print("Label       :", label)
    print("Hand        :", hand)
    print("Shape       :", sequence.shape)
    print("Frames      :", sequence.shape[0])
    print("Features    :", sequence.shape[1])
    print("Data type   :", sequence.dtype)

    print("\nFirst frame:")
    print(sequence[0])

    print("\n===========================================\n")


if __name__ == "__main__":
    main()