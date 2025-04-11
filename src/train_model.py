import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

DATA_DIR = "gesture_data"

all_data = []
labels = []
label_map = {}

for i, file in enumerate(os.listdir(DATA_DIR)):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        label = df['label'][0]
        label_map[i] = label
        labels.extend([i] * len(df))
        df = df.drop(columns=['label'])
        all_data.append(df.values)

X = np.concatenate(all_data)
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

model.save("sign_language_model.h5")
joblib.dump(label_map, "label_map.pkl")

print("Model and label map saved!")