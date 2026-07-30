import cv2

from camera import Camera
from hand_detector import HandDetector
from config import WINDOW_NAME


def main():
    # Initialize camera
    camera = Camera()
    camera.start()

    # Initialize hand detector
    detector = HandDetector()

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


            # ► Testing print statement-remove it later
            


            # Draw landmarks
            frame = detector.draw(frame, results)

            # Display frame
            cv2.imshow(WINDOW_NAME, frame)

            # Exit on Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()


if __name__ == "__main__":
    main()