import csv
import os
import sys
import cv2


from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor


CSV_PATH = "data/processed/keypoint.csv"


def main():

    # Check command-line argument
    if len(sys.argv) != 2:
        print("Usage:")
        print(r"python src\training\dataset_generator.py <LABEL>")
        sys.exit()

    label = sys.argv[1].upper()

    # Create output directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Initialize Camera
    camera = Camera()
    camera.start()

    # Initialize Hand Detector
    detector = HandDetector()

    # Initialize Landmark Processor
    processor = LandmarkProcessor()

    # Open CSV file in append mode
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    try:

        while True:

            frame = camera.get_frame()

            if frame is None:
                break

            # Detect hand
            results = detector.detect(frame)

            # Extract 42 features
            features = processor.extract_features(results)

            # Save sample
            if features:

                row = [label] + features

                writer.writerow(row)

                csv_file.flush()

                print(f"Saved sample for label: {label}")

            # Draw landmarks
            frame = detector.draw(frame, results)

            # Show frame
            cv2.imshow("Dataset Generator", frame)

            # Quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        csv_file.close()

        detector.close()

        camera.stop()


if __name__ == "__main__":
    main()