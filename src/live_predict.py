import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Load trained model
model = tf.keras.models.load_model('sign_language_model.h5')

# Load label map
with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)
label_map = {int(k): v for k, v in label_map.items()}

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# OpenCV camera
cap = cv2.VideoCapture(0)

print("Press Q to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Make sure this matches model's input

            if len(landmarks) == 63:
                input_array = np.array(landmarks).reshape(1, 63)
                prediction = model.predict(input_array)[0]
                class_id = np.argmax(prediction)
                confidence = prediction[class_id]
                label = label_map.get(class_id, "Unknown")

                cv2.putText(frame, f"{label} ({confidence:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

    cv2.imshow("Live Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()