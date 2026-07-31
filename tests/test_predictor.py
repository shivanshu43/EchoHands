from src.core.predictor import Predictor

predictor = Predictor()

dummy_features = [0.0] * 42

predicted_class, confidence = predictor.predict(dummy_features)

print("Predicted Class :", predicted_class)
print("Confidence      :", confidence)