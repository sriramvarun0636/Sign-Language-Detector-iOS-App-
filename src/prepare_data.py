import os
import glob
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

# 📁 Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_DATA_DIR = os.path.join(BASE_DIR, "gesture_data")

print(f"📁 Scanning folder: {GESTURE_DATA_DIR}")
print(f"📂 gesture_data contains: {os.listdir(GESTURE_DATA_DIR)}")

# 📂 Get CSVs
csv_files = glob.glob(os.path.join(GESTURE_DATA_DIR, "*.csv"))
print("✅ CSV files detected:", csv_files)

X = []
y = []

# 📥 Load data
for file_path in csv_files:
    label = os.path.splitext(os.path.basename(file_path))[0]
    print(f"📥 Reading {label} from {file_path}")

    try:
        # ✅ Use genfromtxt to skip headers
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Drop the last column (assuming it's a 'label' column in CSV)
        if data.shape[1] > 63:
            data = data[:, :63]

        X.extend(data)
        y.extend([label] * data.shape[0])

    except Exception as e:
        print(f"⚠️ Skipping {file_path} due to error: {e}")

X = np.array(X)
y = np.array(y)

print(f"📊 Final dataset shape: {X.shape}, Labels: {set(y)}")

# 🧠 Train
clf = DecisionTreeClassifier()
clf.fit(X, y)

# 💾 Save model
model_path = os.path.join(BASE_DIR, "gesture_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"✅ Model saved to {model_path}")