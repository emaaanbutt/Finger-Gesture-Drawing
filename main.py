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

is_drawing = False
is_fist = False


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

                index_tip = hand_landmarks.landmark[8]
                index_dip = hand_landmarks.landmark[7]
                index_pip = hand_landmarks.landmark[5]

                middle_tip = hand_landmarks.landmark[12]
                middle_pip = hand_landmarks.landmark[10]

                ring_tip = hand_landmarks.landmark[16]
                ring_pip = hand_landmarks.landmark[14]

                pinky_tip = hand_landmarks.landmark[20]
                pinky_pip = hand_landmarks.landmark[18]

                ix, iy = int(index_tip.x * w), int(index_tip.y * h)

                
                if index_tip.y < index_dip.y:  
                    is_fist = False

                elif index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y:
                    is_fist = True

                if is_fist:
                    is_drawing = False
                    prev_x, prev_y = None, None 

                else:
                    is_drawing = True

                if is_drawing:
                    if prev_x and prev_y is not None:
                        cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
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