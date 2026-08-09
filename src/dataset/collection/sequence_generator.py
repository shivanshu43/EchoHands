import os
import sys
import time
import cv2
import numpy as np

from src.dataset.collection.collector import DatasetCollector
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor


SEQUENCE_DIR = "data/processed/dynamic_sequences"

SEQUENCES_PER_HAND = 50

MIN_FRAMES = 10

WINDOW_NAME = "Dynamic Sequence Generator"


def save_sequence(sequence, label, hand, sequence_number):

    hand_dir = os.path.join(
        SEQUENCE_DIR,
        label,
        hand,
    )

    os.makedirs(hand_dir, exist_ok=True)

    file_path = os.path.join(
        hand_dir,
        f"{label}_{hand}_{sequence_number:03d}.npz",
    )

    np.savez_compressed(
        file_path,
        sequence=np.asarray(sequence, dtype=np.float32),
        label=label,
        hand=hand,
    )

    return file_path


def draw_dashboard(
    frame,
    label,
    hand,
    status,
    sequence_number,
    total_sequences,
    frame_count,
    message=None,
):

    display_frame = frame.copy()

    cv2.putText(
        display_frame,
        f"Label : {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display_frame,
        f"Hand : {hand}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 180, 0),
        2,
    )

    cv2.putText(
        display_frame,
        f"Sequence : {sequence_number}/{total_sequences}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display_frame,
        f"Frames : {frame_count}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        display_frame,
        f"Status : {status}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
    )

    if message:

        cv2.putText(
            display_frame,
            message,
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

    return display_frame


def wait_for_space(
    collector,
    detector,
    label,
    hand,
    sequence_number,
    total_sequences,
):

    while True:

        frame = collector.get_frame()

        if frame is None:
            continue

        results = detector.detect(frame)

        frame = detector.draw(frame, results)

        display_frame = draw_dashboard(
            frame,
            label,
            hand,
            "READY",
            sequence_number,
            total_sequences,
            0,
            "Press SPACE to START",
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            return

        if key == ord("q"):
            raise KeyboardInterrupt


def record_sequence(
    collector,
    detector,
    processor,
    label,
    hand,
    sequence_number,
    total_sequences,
):

    sequence = []

    while True:

        frame = collector.get_frame()

        if frame is None:
            continue

        results = detector.detect(frame)

        features = processor.extract_features(results)

        if features is not None:

            sequence.append(features)

        frame = detector.draw(frame, results)

        display_frame = draw_dashboard(
            frame,
            label,
            hand,
            "RECORDING",
            sequence_number,
            total_sequences,
            len(sequence),
            "Press SPACE to STOP",
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):

            if len(sequence) < MIN_FRAMES:

                return None, "Sequence too short"

            return sequence, None

        if key == ord("q"):
            raise KeyboardInterrupt


def wait_for_next_sequence(
    collector,
    detector,
    label,
    hand,
    sequence_number,
    total_sequences,
):

    while True:

        frame = collector.get_frame()

        if frame is None:
            continue

        results = detector.detect(frame)

        frame = detector.draw(frame, results)

        display_frame = draw_dashboard(
            frame,
            label,
            hand,
            "SAVED",
            sequence_number,
            total_sequences,
            0,
            "Press SPACE for NEXT sequence",
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            return

        if key == ord("q"):
            raise KeyboardInterrupt


def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python -m "
            "src.dataset.collection.sequence_generator <label>"
        )

        return

    label = sys.argv[1].upper()

    current_hand = "LEFT"

    collector = DatasetCollector()

    detector = HandDetector()

    processor = LandmarkProcessor()

    os.makedirs(
        os.path.join(SEQUENCE_DIR, label),
        exist_ok=True,
    )

    print()
    print("=" * 50)
    print(f"Dynamic Dataset Generator - {label}")
    print("=" * 50)
    print()
    print(f"Sequences per hand : {SEQUENCES_PER_HAND}")
    print(f"Minimum frames     : {MIN_FRAMES}")
    print()
    print("SPACE = Start / Stop")
    print("Q     = Quit")
    print()

    try:

        for hand_index, hand in enumerate(["LEFT", "RIGHT"]):

            current_hand = hand

            print()
            print("=" * 50)
            print(f"Prepare your {current_hand} hand")
            print("=" * 50)
            print()

            for sequence_number in range(
                1,
                SEQUENCES_PER_HAND + 1,
            ):

                # ----------------------------------
                # Wait before recording
                # ----------------------------------

                wait_for_space(
                    collector,
                    detector,
                    label,
                    current_hand,
                    sequence_number,
                    SEQUENCES_PER_HAND,
                )

                # ----------------------------------
                # Record sequence
                # ----------------------------------

                sequence, error = record_sequence(
                    collector,
                    detector,
                    processor,
                    label,
                    current_hand,
                    sequence_number,
                    SEQUENCES_PER_HAND,
                )

                # ----------------------------------
                # Handle short sequence
                # ----------------------------------

                if sequence is None:

                    print(
                        f"\nSequence {sequence_number} "
                        f"was too short."
                    )

                    print(
                        "Please record it again."
                    )

                    continue

                # ----------------------------------
                # Save sequence
                # ----------------------------------

                file_path = save_sequence(
                    sequence,
                    label,
                    current_hand,
                    sequence_number,
                )

                print(
                    f"\nSaved: {file_path}"
                )

                print(
                    f"Frames: {len(sequence)}"
                )

                # ----------------------------------
                # Wait for next sequence
                # ----------------------------------

                if sequence_number < SEQUENCES_PER_HAND:

                    wait_for_next_sequence(
                        collector,
                        detector,
                        label,
                        current_hand,
                        sequence_number,
                        SEQUENCES_PER_HAND,
                    )

            # --------------------------------------
            # Switch hand
            # --------------------------------------

            if hand_index == 0:

                while True:

                    frame = collector.get_frame()

                    if frame is None:
                        continue

                    display_frame = draw_dashboard(
                        frame,
                        label,
                        "RIGHT",
                        "HAND SWITCH",
                        0,
                        SEQUENCES_PER_HAND,
                        0,
                        "Switch to RIGHT hand - "
                        "Press SPACE",
                    )

                    cv2.imshow(
                        WINDOW_NAME,
                        display_frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord(" "):
                        break

                    if key == ord("q"):
                        raise KeyboardInterrupt

        print()
        print("=" * 50)
        print(f"Finished collecting dynamic '{label}'")
        print("=" * 50)

    except KeyboardInterrupt:

        print("\nCollection stopped by user.")

    finally:

        detector.close()
        collector.stop()


if __name__ == "__main__":
    main()