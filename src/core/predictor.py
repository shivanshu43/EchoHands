import numpy as np
import tensorflow as tf

from src.utils.config import MODEL_PATH

class Predictor:

    def __init__(self):

        self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict(self, features):

        if features is None:
            return None

        features = np.array(features, dtype=np.float32)

        features = features.reshape(1, -1)

        prediction = self.model.predict(features, verbose=0)

        predicted_class = np.argmax(prediction)

        confidence = prediction[0][predicted_class]

        return predicted_class, confidence