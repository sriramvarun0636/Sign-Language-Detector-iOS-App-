import numpy as np
import pickle

with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy", allow_pickle=True)

accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")