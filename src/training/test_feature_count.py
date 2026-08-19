from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor


def main():

    camera = Camera()
    detector = HandDetector()
    processor = LandmarkProcessor()

    camera.start()

    print("\nShow your hand to the camera.")
    print("Press Q to quit.\n")

    try:

        while True:

            frame = camera.get_frame()

            if frame is None:
                break

            results = detector.detect(frame)

            features = processor.extract_features(
                results
            )

            if features is not None:

                print(
                    f"\rFeature count: "
                    f"{len(features)}",
                    end=""
                )

            frame = detector.draw(
                frame,
                results
            )

            import cv2

            cv2.imshow(
                "Feature Test",
                frame
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

    finally:

        detector.close()
        camera.stop()


if __name__ == "__main__":
    main()