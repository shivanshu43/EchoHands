class LandmarkProcessor:

    def extract_features(self, results):

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]

        features = []

        for landmark in hand.landmark:

            features.append(landmark.x)
            features.append(landmark.y)

        return features