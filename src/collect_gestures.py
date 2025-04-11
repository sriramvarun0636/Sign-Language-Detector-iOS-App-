import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

DATA_DIR = 'gesture_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

label = input("Enter gesture label (e.g. Hello, Yes): ").strip().capitalize()
cap = cv2.VideoCapture(0)
data = []

print("Press 'S' to save frame, 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)

    if key == ord('s') and results.multi_hand_landmarks:
        data.append(landmarks)
        print(f"Saved frame #{len(data)}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save data only if something was collected
if data:
    df = pd.DataFrame(data)
    df['label'] = label
    df.to_csv(f"{DATA_DIR}/{label}.csv", index=False)
    print(f"Saved {len(data)} samples to {DATA_DIR}/{label}.csv")
else:
    print("No data collected. Nothing saved.")