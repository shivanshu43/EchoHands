import csv
import os
import sys
import time
import cv2

from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor

CSV_PATH = "data/processed/keypoint.csv"


def main():

    if len(sys.argv) != 2:
        print("Usage: python -m src.training.dataset_generator <label>")
        return

    label = sys.argv[1].upper()

    TARGET_SAMPLES = 300
    sample_count = 0

    CAPTURE_INTERVAL = 0.25  # seconds

    last_capture_time = time.time()

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    camera = Camera()
    camera.start()

    detector = HandDetector()
    processor = LandmarkProcessor()

    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    print(f"\nCollecting samples for label '{label}'")
    print("Press Q to quit.\n")

    try:

        while True:

            frame = camera.get_frame()

            if frame is None:
                print("Failed to capture frame.")
                break

            results = detector.detect(frame)

            features = processor.extract_features(results)

            current_time = time.time()

            if features and current_time - last_capture_time >= CAPTURE_INTERVAL:

                row = [label] + features

                writer.writerow(row)
                csv_file.flush()

                last_capture_time = current_time

                sample_count += 1

                print(
                    f"\rSamples: {sample_count}/{TARGET_SAMPLES}",
                    end=""
                )

                if sample_count >= TARGET_SAMPLES:

                    print("\nDataset collection completed!")

                    break

            frame = detector.draw(frame, results)

            cv2.putText(
                frame,
                f"Samples : {sample_count}/{TARGET_SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Dataset Generator", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        csv_file.close()
        detector.close()
        camera.stop()


if __name__ == "__main__":
    main()