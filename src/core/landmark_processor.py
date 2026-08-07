import math


class LandmarkProcessor:

    def extract_features(self, results):

        if results is None or not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]

        wrist = hand.landmark[0]

        wrist_x = wrist.x
        wrist_y = wrist.y

        # ---------------------------------
        # Find maximum distance from wrist
        # ---------------------------------

        max_distance = 0.0

        for landmark in hand.landmark:

            dx = landmark.x - wrist_x
            dy = landmark.y - wrist_y

            distance = math.sqrt(dx * dx + dy * dy)

            if distance > max_distance:
                max_distance = distance

        if max_distance == 0:
            max_distance = 1

        # ---------------------------------
        # Build normalized feature vector
        # ---------------------------------

        features = []

        for landmark in hand.landmark:

            features.append((landmark.x - wrist_x) / max_distance)
            features.append((landmark.y - wrist_y) / max_distance)

        return features