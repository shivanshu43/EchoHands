import math


class QualityChecker:

    # Validates whether a detected hand is suitable
    # for adding to the dataset.

    # The checker focuses on landmark quality rather
    # than the shape/orientation of the gesture.

    # This is important because alphabet gestures may
    # contain open, closed, folded, tilted, or rotated
    # fingers.
    

    def is_valid(self, results):

        # ==================================================
        # 1. Hand must be detected
        # ==================================================

        if results is None:
            return False

        if not results.multi_hand_landmarks:
            return False

        # ==================================================
        # 2. Only one hand should be present
        # ==================================================

        if len(results.multi_hand_landmarks) != 1:
            return False

        # Get the detected hand
        hand = results.multi_hand_landmarks[0]

        landmarks = hand.landmark

        # ==================================================
        # 3. MediaPipe should provide all 21 landmarks
        # ==================================================

        if len(landmarks) != 21:
            return False

        # ==================================================
        # 4. Check that landmark coordinates are valid
        # ==================================================

        for landmark in landmarks:

            if not math.isfinite(landmark.x):
                return False

            if not math.isfinite(landmark.y):
                return False

            if not math.isfinite(landmark.z):
                return False

        # ==================================================
        # 5. Check that the wrist exists inside the frame
        # ==================================================

        wrist = landmarks[0]

        if wrist.x < 0.0 or wrist.x > 1.0:
            return False

        if wrist.y < 0.0 or wrist.y > 1.0:
            return False

        # ==================================================
        # 6. Check overall landmark spread
        #
        # We only reject extremely tiny detections.
        #
        # We DO NOT require the hand to be open.
        # Therefore folded fingers in M/N are allowed.
        # ==================================================

        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]

        min_x = min(xs)
        max_x = max(xs)

        min_y = min(ys)
        max_y = max(ys)

        width = max_x - min_x
        height = max_y - min_y

        # Extremely tiny hand detection usually means
        # MediaPipe has produced an unreliable landmark set.
        if width < 0.05 and height < 0.05:
            return False

        # ==================================================
        # 7. Check that landmarks are not completely
        # collapsed into one point
        # ==================================================

        wrist_x = wrist.x
        wrist_y = wrist.y

        landmark_spread = 0.0

        for landmark in landmarks:

            dx = landmark.x - wrist_x
            dy = landmark.y - wrist_y

            distance = math.sqrt(
                dx * dx +
                dy * dy
            )

            landmark_spread = max(
                landmark_spread,
                distance
            )

        # A real hand should have some spatial spread.
        if landmark_spread < 0.05:
            return False

        # ==================================================
        # 8. Valid sample
        # ==================================================

        return True