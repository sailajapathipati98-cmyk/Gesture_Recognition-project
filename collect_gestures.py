# collect_gestures.py
import cv2
import mediapipe as mp
import numpy as np
import csv
import os

GESTURES = [
    'Fist','Palm','ThumbsUp','ThumbsDown','Peace','OK','CallMe','Rock','Index','Spider',
    'Heart','Stop','Gun','Three','Four','Five','Six','Seven','Nine','Victory'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

file_name = 'gesture_data.csv'
if not os.path.exists(file_name):
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        # header can be optional
        writer.writerow([f'lm{i}' for i in range(63)] + ['label'])

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    input(f"Press Enter and show gesture: {gesture}")
    count = 0
    while count < 50:  # capture 50 frames per gesture
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                with open(file_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks + [gesture])
                count += 1

        cv2.putText(frame, f'{gesture} - {count}/50', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Gesture data collection complete!")
