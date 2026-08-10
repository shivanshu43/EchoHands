from src.core.sequence_detector import SequenceDetector


class RecognitionController:

    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    COOLDOWN = "COOLDOWN"

    def __init__(
        self,
        static_predictor,
        dynamic_predictor,
        cooldown_frames=40,
        dynamic_confidence_threshold=0.85,
    ):

        self.static_predictor = static_predictor
        self.dynamic_predictor = dynamic_predictor

        self.sequence_detector = SequenceDetector()

        self.mode = self.STATIC

        self.last_prediction = None
        self.last_confidence = 0.0

        # ------------------------------------------
        # Dynamic gesture cooldown
        # ------------------------------------------

        self.cooldown_frames = cooldown_frames

        self.dynamic_confidence_threshold = (
            dynamic_confidence_threshold
        )

        self.cooldown_counter = 0

    def update(self, features):

        # ==========================================
        # NO HAND
        # ==========================================

        if features is None:

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.mode,
                "sequence_complete": False,
            }

        # ==========================================
        # COOLDOWN
        # ==========================================

        if self.mode == self.COOLDOWN:

            self.cooldown_counter += 1

            # --------------------------------------
            # Still inside cooldown
            # --------------------------------------

            if (
                self.cooldown_counter
                < self.cooldown_frames
            ):

                return {
                    "prediction": self.last_prediction,
                    "confidence": self.last_confidence,
                    "mode": self.COOLDOWN,
                    "sequence_complete": False,
                }

            # --------------------------------------
            # Cooldown finished
            # --------------------------------------

            self.mode = self.STATIC
            self.cooldown_counter = 0

        # ==========================================
        # Dynamic sequence detection
        # ==========================================

        completed_sequence = (
            self.sequence_detector.update(
                features
            )
        )

        # ==========================================
        # Dynamic sequence currently recording
        # ==========================================

        if (
            self.sequence_detector.get_state()
            == SequenceDetector.RECORDING
        ):

            self.mode = self.DYNAMIC

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.DYNAMIC,
                "sequence_complete": False,
            }

        # ==========================================
        # Dynamic gesture completed
        # ==========================================

        if completed_sequence is not None:

            prediction, confidence = (
                self.dynamic_predictor.predict(
                    completed_sequence
                )
            )

            # --------------------------------------
            # Check dynamic confidence
            # --------------------------------------

            if (
                prediction is not None
                and confidence
                >= self.dynamic_confidence_threshold
            ):

                # Valid dynamic gesture

                self.last_prediction = prediction
                self.last_confidence = confidence

                self.mode = self.COOLDOWN
                self.cooldown_counter = 0

                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "mode": self.DYNAMIC,
                    "sequence_complete": True,
                }

            # --------------------------------------
            # Low-confidence sequence
            # --------------------------------------

            # Treat it as accidental movement.

            self.mode = self.STATIC
            self.cooldown_counter = 0

            return {
                "prediction": None,
                "confidence": 0.0,
                "mode": self.STATIC,
                "sequence_complete": False,
            }

        # ==========================================
        # Static recognition
        # ==========================================

        self.mode = self.STATIC

        prediction, confidence = (
            self.static_predictor.predict(
                features
            )
        )

        self.last_prediction = prediction
        self.last_confidence = confidence

        return {
            "prediction": prediction,
            "confidence": confidence,
            "mode": self.STATIC,
            "sequence_complete": False,
        }

    def get_mode(self):

        return self.mode

    def get_sequence_length(self):

        return (
            self.sequence_detector
            .get_sequence_length()
        )

    def get_cooldown_progress(self):

        return self.cooldown_counter

    def reset(self):

        self.sequence_detector.reset()

        self.mode = self.STATIC

        self.last_prediction = None
        self.last_confidence = 0.0

        self.cooldown_counter = 0