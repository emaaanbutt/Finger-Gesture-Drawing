import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

smooth_x, smooth_y = None, None
alpha = 0.25 

canvas = np.zeros_like(frame)
prev_x, prev_y = None, None 


with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, c = frame.shape

                index_finger = hand_landmarks.landmark[8]

                ix, iy = int(index_finger.x * w), int(index_finger.y * h)

                cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)

                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0,0,255), 3)
                
                prev_x, prev_y = ix, iy

        # combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)

        frame [mask > 0] = canvas[mask > 0]

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()