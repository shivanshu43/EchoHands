import numpy as np

from src.core.sequence_detector import SequenceDetector


class RecognitionController:

    NONE = "NONE"
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"

    def __init__(
        self,
        static_predictor,
        dynamic_predictor,
        dynamic_confidence_threshold=0.85,
        movement_threshold=0.035,
        unlock_threshold=0.035,
        static_prediction_frames=4,
    ):

        self.static_predictor = static_predictor
        self.dynamic_predictor = dynamic_predictor

        self.sequence_detector = SequenceDetector()

        # ==========================================
        # Thresholds
        # ==========================================

        self.dynamic_confidence_threshold = (
            dynamic_confidence_threshold
        )

        # Movement required to start a new gesture
        self.movement_threshold = movement_threshold

        # Difference from the previously recognized
        # gesture required to unlock NONE state
        self.unlock_threshold = unlock_threshold

        # Number of consecutive frames required
        # before accepting a static prediction
        self.static_prediction_frames = (
            static_prediction_frames
        )

        # ==========================================
        # State
        # ==========================================

        self.mode = self.NONE

        self.previous_features = None

        # ==========================================
        # Gesture Lock
        # ==========================================

        # Stores the feature vector of the gesture
        # that was most recently recognized.
        #
        # While the user keeps holding that gesture,
        # the controller remains in NONE.
        self.locked_features = None

        # ==========================================
        # Static recognition tracking
        # ==========================================

        self.static_candidate = None
        self.static_candidate_count = 0

        # ==========================================
        # Last prediction
        # ==========================================

        self.last_prediction = None
        self.last_confidence = 0.0

        self.gesture_emitted = False

    # ==========================================================
    # Motion calculation
    # ==========================================================

    def _calculate_motion(self, features):

        if (
            features is None
            or self.previous_features is None
        ):

            return 0.0

        current = np.asarray(
            features,
            dtype=np.float32
        )

        previous = np.asarray(
            self.previous_features,
            dtype=np.float32
        )

        return float(
            np.mean(
                np.abs(
                    current - previous
                )
            )
        )

    # ==========================================================
    # Change from locked gesture
    # ==========================================================

    def _calculate_change_from_locked(self, features):

        if (
            features is None
            or self.locked_features is None
        ):

            return 0.0

        current = np.asarray(
            features,
            dtype=np.float32
        )

        locked = np.asarray(
            self.locked_features,
            dtype=np.float32
        )

        return float(
            np.mean(
                np.abs(
                    current - locked
                )
            )
        )

    # ==========================================================
    # Reset current recognition attempt
    # ==========================================================

    def _reset_gesture_state(self):

        self.sequence_detector.reset()

        self.static_candidate = None
        self.static_candidate_count = 0

        self.gesture_emitted = False

    # ==========================================================
    # Update
    # ==========================================================

    def update(self, features):

        # ======================================================
        # NO HAND
        # ======================================================

        if features is None:

            self.previous_features = None

            self._reset_gesture_state()

            # If the hand disappears completely,
            # remove the previous gesture lock.
            self.locked_features = None

            self.mode = self.NONE

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.mode,
                "sequence_complete": False,
            }

        # ======================================================
        # Calculate frame-to-frame motion
        # ======================================================

        motion = self._calculate_motion(
            features
        )

        # ======================================================
        # NONE
        #
        # This is the gesture-lock state.
        #
        # If a gesture was already recognized:
        #
        #     recognized gesture
        #             ↓
        #           NONE
        #             ↓
        #       hold same pose
        #             ↓
        #           NONE
        #
        # Only when the pose changes enough do we unlock
        # and allow another gesture to be recognized.
        # ======================================================

        if self.mode == self.NONE:

            # --------------------------------------------------
            # No gesture is currently locked
            # --------------------------------------------------

            if self.locked_features is None:

                # First frame after reset
                if self.previous_features is None:

                    self.previous_features = features

                    return {
                        "prediction": None,
                        "confidence": 0.0,
                        "mode": self.NONE,
                        "sequence_complete": False,
                    }

                # ----------------------------------------------
                # Movement detected
                # ----------------------------------------------

                if motion >= self.movement_threshold:

                    self.mode = self.STATIC

                else:

                    self.previous_features = features

                    return {
                        "prediction": None,
                        "confidence": 0.0,
                        "mode": self.NONE,
                        "sequence_complete": False,
                    }

            # --------------------------------------------------
            # A gesture is currently locked
            # --------------------------------------------------

            else:

                gesture_change = (
                    self._calculate_change_from_locked(
                        features
                    )
                )

                # ----------------------------------------------
                # Same gesture
                # ----------------------------------------------

                if (
                    gesture_change
                    < self.unlock_threshold
                ):

                    self.previous_features = features

                    return {
                        "prediction": None,
                        "confidence": 0.0,
                        "mode": self.NONE,
                        "sequence_complete": False,
                    }

                # ----------------------------------------------
                # Gesture changed
                # ----------------------------------------------

                self.locked_features = None

                self._reset_gesture_state()

                self.mode = self.STATIC

            self.previous_features = features

        # ======================================================
        # STATIC
        #
        # Static recognition is active here.
        #
        # If movement becomes a dynamic sequence,
        # switch to DYNAMIC.
        # ======================================================

        if self.mode == self.STATIC:

            # --------------------------------------------------
            # Check for dynamic movement
            # --------------------------------------------------

            if motion >= self.movement_threshold:

                self.sequence_detector.update(
                    features
                )

                if (
                    self.sequence_detector.get_state()
                    == SequenceDetector.RECORDING
                ):

                    self.mode = self.DYNAMIC

                    self.static_candidate = None
                    self.static_candidate_count = 0

                    self.previous_features = features

                    return {
                        "prediction": None,
                        "confidence": 0.0,
                        "mode": self.DYNAMIC,
                        "sequence_complete": False,
                    }

            # --------------------------------------------------
            # Static prediction
            # --------------------------------------------------

            prediction, confidence = (
                self.static_predictor.predict(
                    features
                )
            )

            # --------------------------------------------------
            # Track stable prediction
            # --------------------------------------------------

            if prediction == self.static_candidate:

                self.static_candidate_count += 1

            else:

                self.static_candidate = prediction
                self.static_candidate_count = 1

            self.last_prediction = prediction
            self.last_confidence = confidence

            # --------------------------------------------------
            # Static gesture recognized
            # --------------------------------------------------

            if (
                self.static_candidate_count
                >= self.static_prediction_frames
                and confidence
                >= 0.50
            ):

                self.gesture_emitted = True

                # ----------------------------------------------
                # LOCK THE CURRENT POSE
                # ----------------------------------------------

                self.locked_features = np.asarray(
                    features,
                    dtype=np.float32
                ).copy()

                # ----------------------------------------------
                # Immediately enter NONE
                # ----------------------------------------------

                self.mode = self.NONE

                self.previous_features = features

                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "mode": self.NONE,
                    "sequence_complete": False,
                }

            # --------------------------------------------------
            # Static gesture still being evaluated
            # --------------------------------------------------

            self.previous_features = features

            return {
                "prediction": prediction,
                "confidence": confidence,
                "mode": self.STATIC,
                "sequence_complete": False,
            }

        # ======================================================
        # DYNAMIC
        # ======================================================

        if self.mode == self.DYNAMIC:

            completed_sequence = (
                self.sequence_detector.update(
                    features
                )
            )

            # --------------------------------------------------
            # Sequence still recording
            # --------------------------------------------------

            if completed_sequence is None:

                self.previous_features = features

                return {
                    "prediction": None,
                    "confidence": 0.0,
                    "mode": self.DYNAMIC,
                    "sequence_complete": False,
                }

            # --------------------------------------------------
            # Dynamic sequence completed
            # --------------------------------------------------

            prediction, confidence = (
                self.dynamic_predictor.predict(
                    completed_sequence
                )
            )

            # --------------------------------------------------
            # Valid dynamic gesture
            # --------------------------------------------------

            if (
                prediction is not None
                and confidence
                >= self.dynamic_confidence_threshold
            ):

                self.last_prediction = prediction
                self.last_confidence = confidence

                self.gesture_emitted = True

                # ----------------------------------------------
                # LOCK FINAL POSE
                # ----------------------------------------------

                self.locked_features = np.asarray(
                    features,
                    dtype=np.float32
                ).copy()

                # ----------------------------------------------
                # Immediately enter NONE
                # ----------------------------------------------

                self.mode = self.NONE

                self.previous_features = features

                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "mode": self.NONE,
                    "sequence_complete": True,
                }

            # --------------------------------------------------
            # Invalid dynamic sequence
            # --------------------------------------------------

            self._reset_gesture_state()

            self.locked_features = None

            self.mode = self.NONE

            self.previous_features = features

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.NONE,
                "sequence_complete": False,
            }

        # ======================================================
        # FALLBACK
        # ======================================================

        self._reset_gesture_state()

        self.locked_features = None

        self.mode = self.NONE

        self.previous_features = features

        return {
            "prediction": None,
            "confidence": 0.0,
            "mode": self.NONE,
            "sequence_complete": False,
        }

    # ==========================================================
    # Getters
    # ==========================================================

    def get_mode(self):

        return self.mode

    def get_sequence_length(self):

        return (
            self.sequence_detector
            .get_sequence_length()
        )

    def reset(self):

        self.sequence_detector.reset()

        self.mode = self.NONE

        self.previous_features = None

        self.locked_features = None

        self.static_candidate = None
        self.static_candidate_count = 0

        self.last_prediction = None
        self.last_confidence = 0.0

        self.gesture_emitted = False