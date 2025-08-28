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
is_palm = False
color = (0, 0, 255)
color_name = "Red"


with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        cv2.putText(frame, "1 for red", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "2 for green", (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "3 for blue", (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, c = frame.shape

                index_tip = hand_landmarks.landmark[8]
                index_dip = hand_landmarks.landmark[7]
                index_pip = hand_landmarks.landmark[5]

                middle_tip = hand_landmarks.landmark[12]
                middle_dip = hand_landmarks.landmark[11]
                middle_pip = hand_landmarks.landmark[10]

                ring_tip = hand_landmarks.landmark[16]
                ring_dip = hand_landmarks.landmark[15]
                ring_pip = hand_landmarks.landmark[14]

                pinky_tip = hand_landmarks.landmark[20]
                pinky_dip = hand_landmarks.landmark[19]
                pinky_pip = hand_landmarks.landmark[18]

                ix, iy = int(index_tip.x * w), int(index_tip.y * h)

                if index_tip.y < index_dip.y  and middle_tip.y > middle_dip.y and ring_tip.y > ring_dip.y and pinky_tip.y > pinky_dip.y:
                    color = (0, 0, 255)
                    color_name = "Red"
                elif index_tip.y < index_dip.y  and middle_tip.y < middle_dip.y and ring_tip.y > ring_dip.y and pinky_tip.y > pinky_dip.y:
                    color = (0, 255, 0)
                    color_name = "Green"
                elif index_tip.y < index_dip.y  and middle_tip.y < middle_dip.y and ring_tip.y < ring_dip.y and pinky_tip.y > pinky_dip.y:
                    color = (255, 0, 0)
                    color_name = "Blue"

                cv2.putText(frame, f"Color selected: {color_name}", (380, 450), cv2.FONT_HERSHEY_SIMPLEX,  0.75, (0, 0, 255), 2, cv2.LINE_AA)

                if index_tip.y < index_dip.y and middle_tip.y < middle_dip.y and ring_tip.y < ring_dip.y and pinky_tip.y < pinky_dip.y:
                    is_palm = True
                    is_fist =  False
                elif (index_tip.y < index_dip.y or middle_tip.y < middle_dip.y or ring_tip.y < ring_dip.y) and pinky_tip.y > pinky_dip.y: 
                    is_fist = False
                elif index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y:
                    is_fist = True

                if is_fist:
                    is_drawing = False
                    prev_x, prev_y = None, None 
                elif is_fist is False:
                    is_drawing = True

                if is_palm:
                    canvas = np.zeros_like(frame)

                if is_drawing:
                    if prev_x and prev_y is not None:
                        cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
                        cv2.line(canvas, (prev_x, prev_y), (ix, iy), color, 3)

                if is_palm:
                    cv2.putText(frame, 'Clear', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif is_drawing:
                    cv2.putText(frame, 'Drawing mode: On', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Drawing mode: Off', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                prev_x, prev_y = ix, iy
                # is_drawing = False
                is_palm = False

        # combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)

        frame [mask > 0] = canvas[mask > 0]

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()