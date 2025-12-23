import cv2
import mediapipe as mp
import time
from collections import deque

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ================= FINGER LOGIC ==============
def count_fingers(hand_landmarks, hand_label):
    fingers = 0
    lm = hand_landmarks.landmark

    # Thumb (orientation aware)
    if hand_label == "Right":
        if lm[4].x > lm[3].x:
            fingers += 1
    else:
        if lm[4].x < lm[3].x:
            fingers += 1

    # Other fingers
    for tip in [8, 12, 16, 20]:
        if lm[tip].y < lm[tip - 2].y:
            fingers += 1

    return fingers

# ================= STABILITY ==================
finger_buffer = deque(maxlen=7)

last_value = -1
stable_value = -1
last_change_time = time.time()
STABLE_TIME = 0.7

# ================= HUD ==================
BAR_WIDTH = 300
BAR_HEIGHT = 20
BAR_X = 20
BAR_Y = 90

# ================= MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hud_color = (0, 255, 255)  # yellow (detecting)
    progress = 0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            hand_label = results.multi_handedness[idx].classification[0].label

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = count_fingers(hand_landmarks, hand_label)
            finger_buffer.append(fingers)

            smooth_fingers = max(set(finger_buffer), key=finger_buffer.count)

            if smooth_fingers != last_value:
                last_value = smooth_fingers
                last_change_time = time.time()

            elapsed = time.time() - last_change_time
            progress = min(elapsed / STABLE_TIME, 1.0)

            if elapsed > STABLE_TIME:
                stable_value = smooth_fingers
                hud_color = (0, 255, 0)  # green (stable)

            # -------- HUD TEXT --------
            cv2.putText(
                frame,
                f"{smooth_fingers}",
                (w // 2 - 40, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                hud_color,
                6
            )

            cv2.putText(
                frame,
                f"Hand: {hand_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

    # -------- PROGRESS BAR --------
    cv2.rectangle(
        frame,
        (BAR_X, BAR_Y),
        (BAR_X + BAR_WIDTH, BAR_Y + BAR_HEIGHT),
        (255, 255, 255),
        2
    )

    cv2.rectangle(
        frame,
        (BAR_X, BAR_Y),
        (BAR_X + int(BAR_WIDTH * progress), BAR_Y + BAR_HEIGHT),
        hud_color,
        -1
    )

    cv2.imshow("Hand Gesture Detection - Visual HUD", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
