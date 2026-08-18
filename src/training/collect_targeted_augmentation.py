import csv
import os
import sys
import time
import cv2

from src.dataset.collection.collector import DatasetCollector
from src.dataset.collection.quality_checker import QualityChecker
from src.dataset.collection.duplicate_detector import DuplicateDetector

from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor


CSV_PATH = "data/processed/keypoints.csv"

TARGET_NEW_SAMPLES = 100

SAMPLES_PER_VARIATION = 20

VARIATIONS = [
    "Canonical Pose",
    "Slight Wrist Rotation Left",
    "Slight Wrist Rotation Right",
    "Natural Finger Configuration",
    "Slight Palm Tilt",
]

CAPTURE_INTERVAL = 0.25


def get_existing_samples(csv_path, label):

    if not os.path.exists(csv_path):
        return 0

    count = 0

    with open(csv_path, "r", newline="") as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) > 0 and row[0] == label:
                count += 1

    return count


def collect_label(label):

    existing = get_existing_samples(
        CSV_PATH,
        label
    )

    print()
    print("=" * 55)
    print(f"Targeted Augmentation: {label}")
    print("=" * 55)
    print(f"Existing samples : {existing}")
    print(f"New samples      : {TARGET_NEW_SAMPLES}")
    print(f"Final total      : {existing + TARGET_NEW_SAMPLES}")
    print("=" * 55)
    print()

    collector = DatasetCollector()
    detector = HandDetector()
    processor = LandmarkProcessor()
    quality_checker = QualityChecker()
    duplicate_detector = DuplicateDetector()

    variation_index = 0
    variation_count = 0

    new_samples = 0

    paused = True

    pause_message = (
        f"Prepare {label} - "
        f"{VARIATIONS[variation_index]}"
    )

    last_capture_time = time.time()

    os.makedirs(
        os.path.dirname(CSV_PATH),
        exist_ok=True
    )

    csv_file = open(
        CSV_PATH,
        "a",
        newline=""
    )

    writer = csv.writer(csv_file)

    try:

        while True:

            frame = collector.get_frame()

            if frame is None:
                print("\nFailed to capture frame.")
                break

            results = detector.detect(frame)

            features = None

            # ======================================
            # Quality Check
            # ======================================

            if quality_checker.is_valid(results):

                features = processor.extract_features(
                    results
                )

                # ==================================
                # Duplicate Check
                # ==================================

                if duplicate_detector.is_duplicate(
                    features
                ):

                    features = None

            current_time = time.time()

            # ======================================
            # Save Sample
            # ======================================

            if (
                not paused
                and features is not None
                and current_time - last_capture_time
                >= CAPTURE_INTERVAL
            ):

                row = [label] + features

                writer.writerow(row)

                csv_file.flush()

                new_samples += 1
                variation_count += 1

                last_capture_time = current_time

                print(
                    f"\r{label}: "
                    f"{new_samples}/{TARGET_NEW_SAMPLES}",
                    end=""
                )

                # ==================================
                # Move to next variation
                # ==================================

                if (
                    variation_count
                    >= SAMPLES_PER_VARIATION
                ):

                    variation_count = 0

                    variation_index += 1

                    # ------------------------------
                    # All variations completed
                    # ------------------------------

                    if (
                        variation_index
                        >= len(VARIATIONS)
                    ):

                        variation_index = (
                            len(VARIATIONS) - 1
                        )

                    paused = True

                    pause_message = (
                        f"Next: {label} - "
                        f"{VARIATIONS[variation_index]}"
                    )

            # ======================================
            # Draw landmarks
            # ======================================

            frame = detector.draw(
                frame,
                results
            )

            # ======================================
            # Dashboard
            # ======================================

            cv2.putText(
                frame,
                f"Label: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"New Samples: "
                f"{new_samples}/{TARGET_NEW_SAMPLES}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Variation: "
                f"{VARIATIONS[variation_index]}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            if paused:

                cv2.putText(
                    frame,
                    pause_message,
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Press SPACE to start",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Targeted Static Augmentation",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # ======================================
            # Quit
            # ======================================

            if key == ord("q"):

                break

            # ======================================
            # Resume capture
            # ======================================

            if paused and key == ord(" "):

                paused = False

                pause_message = ""

            # ======================================
            # Finished
            # ======================================

            if new_samples >= TARGET_NEW_SAMPLES:

                print()
                print()
                print(
                    f"Finished targeted "
                    f"augmentation for {label}."
                )

                break

    finally:

        csv_file.close()

        detector.close()

        collector.stop()

        cv2.destroyAllWindows()


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: "
            "python -m src.training."
            "collect_targeted_augmentation "
            "<label>"
        )

        return

    label = sys.argv[1].upper()

    if len(label) != 1:

        print("Label must be a single character.")

        return

    collect_label(label)


if __name__ == "__main__":

    main()