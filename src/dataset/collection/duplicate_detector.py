import numpy as np


class DuplicateDetector:
    
    # Detects whether the current sample is too similar
    # to the previously saved sample
    

    def __init__(self, threshold=0.01):

        self.threshold = threshold

        self.previous_sample = None

    def is_duplicate(self, features):

        if features is None:
            return True

        if self.previous_sample is None:

            self.previous_sample = np.array(features)

            return False

        current = np.array(features)

        previous = np.array(self.previous_sample)

        difference = np.mean(np.abs(current - previous))

        if difference < self.threshold:

            return True

        self.previous_sample = features

        return False