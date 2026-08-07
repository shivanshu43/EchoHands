from src.training.dataset_loader import load_dataset

X_train, X_test, y_train, y_test, encoder = load_dataset()

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

print("Classes :", encoder.classes_)