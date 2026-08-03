class QualityChecker:
    
    # Validates whether a detected hand is suitable
    # for adding to the dataset
    

    def is_valid(self, results):

        # No hand detected
        if results is None or not results.multi_hand_landmarks:
            return False

        # More rules will be added later

        return True