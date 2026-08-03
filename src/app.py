import cv2

from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor
from src.core.predictor import Predictor

from src.utils.config import WINDOW_NAME


def main():

    # Initialize camera
    camera = Camera()
    camera.start()

    # Initialize hand detector
    detector = HandDetector()

    # Initialize landmark processor
    processor = LandmarkProcessor()
    predictor = Predictor()

    print("Press 'Q' to exit.")

    try:
        while True:

            # Get frame from webcam
            frame = camera.get_frame()

            if frame is None:
                print("Failed to capture frame.")
                break

            # Detect hands
            results = detector.detect(frame)

            # Extract 63 features
            features = processor.extract_features(results)



            # ► Testing - remove later
            prediction_text = "No Hand Detected"

            if features is not None:

             predicted_class, confidence = predictor.predict(features)

             prediction_text = (
                 f"Class: {predicted_class} | "
                  f"Confidence: {confidence * 100:.1f}%"
              )



            # Draw landmarks
            frame = detector.draw(frame, results)

            cv2.putText(
                 frame,
                 prediction_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # Display frame
            cv2.imshow(WINDOW_NAME, frame)

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        detector.close()
        camera.stop()


if __name__ == "__main__":
    main()