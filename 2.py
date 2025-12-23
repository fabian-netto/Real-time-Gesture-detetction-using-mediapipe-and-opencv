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

# ================= FINGER COUNT =================
def count_fingers(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    fingers = 0

    # Thumb
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

# ================= SMOOTHING =================
finger_buffer = deque(maxlen=7)

last_value = -1
stable_value = -1
last_change_time = time.time()
STABLE_TIME = 0.7

# ================= ACTION COOLDOWN =================
last_action_time = 0
ACTION_COOLDOWN = 1.5  # seconds

# ================= ACTION MAP =================
ACTION_MAP = {
    0: "PAUSE",
    1: "VOLUME UP",
    2: "VOLUME DOWN",
    3: "PLAY",
    4: "NEXT",
    5: "EXIT"
}

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hud_color = (0, 255, 255)
    progress = 0
    active_command = "NONE"

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
                hud_color = (0, 255, 0)

                now = time.time()
                if now - last_action_time > ACTION_COOLDOWN:
                    if stable_value in ACTION_MAP:
                        active_command = ACTION_MAP[stable_value]
                        last_action_time = now

                        # ---- ACTION EXECUTION ----
                        print(f"COMMAND EXECUTED: {active_command}")

                        if active_command == "EXIT":
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()

            # -------- HUD DISPLAY --------
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
                f"Command: {active_command}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

    # -------- STABILITY BAR --------
    cv2.rectangle(frame, (20, 80), (320, 100), (255, 255, 255), 2)
    cv2.rectangle(frame, (20, 80), (20 + int(300 * progress), 100), hud_color, -1)

    cv2.imshow("Gesture Controlled System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
