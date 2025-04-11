import numpy as np
import os

GESTURES = ["Hello", "Me", "No", "Peace", "Yes"]
X, y = [], []

for idx, gesture in enumerate(GESTURES):
    data = np.load(f"data/{gesture}.npy")
    X.append(data)
    y.append(np.full(len(data), idx))

X = np.concatenate(X)
y = np.concatenate(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("✅ Dataset prepared and saved as X.npy and y.npy.")