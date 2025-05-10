import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load model
model = load_model("eye_state_model.h5")
IMG_SIZE = 96
label_dict = {0: "Closed", 1: "Open"}

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Input timer
focus_minutes = int(input("Enter your focus time (min): "))
focus_seconds = focus_minutes * 60
start_time = time.time()

# Status counters
required_frames = 8
focus_counter = 0
unfocus_counter = 0
current_status = "FOCUS"

# Fallback timer setup
last_open_time = time.time()
fallback_timeout = 10  # seconds

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eye_states = []

    for (fx, fy, fw, fh) in faces:
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Reject if too wide or too low (likely mouth)
            aspect_ratio = ew / float(eh)
            eye_center_y = fy + ey + eh / 2
            if aspect_ratio > 2.5 or eye_center_y > fy + fh / 2:
                continue

            x1 = fx + ex
            y1 = fy + ey
            x2 = x1 + ew
            y2 = y1 + eh

            eye_roi = gray[y1:y2, x1:x2]
            if eye_roi.shape[0] < 10 or eye_roi.shape[1] < 10:
                continue

            eye_resized = cv2.resize(eye_roi, (IMG_SIZE, IMG_SIZE))
            eye_input = eye_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

            pred = model.predict(eye_input, verbose=0)[0]
            confidence = np.max(pred)
            if confidence < 0.7:
                continue

            class_idx = np.argmax(pred)
            eye_states.append(class_idx)

            label = label_dict[class_idx]
            color = (0, 255, 0) if class_idx == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Decide current status
    if len(eye_states) == 2:
        if any(state == 0 for state in eye_states):  # one or both closed
            unfocus_counter += 1
            focus_counter = 0
        else:  # both open
            focus_counter += 1
            unfocus_counter = 0
            last_open_time = time.time()
    elif len(eye_states) == 1:
        # only one eye detected = too risky to call focused
        unfocus_counter += 1
        focus_counter = 0

    if unfocus_counter >= required_frames:
        current_status = "UNFOCUS"
    if focus_counter >= required_frames:
        current_status = "FOCUS"

    # no open eyes detected for X seconds
    if time.time() - last_open_time > fallback_timeout:
        current_status = "UNFOCUS"
        cv2.putText(frame, "⚠️ Eyes not detected for 10s!", (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Write current status to file (for chatbot to read)
    with open("focus_status.txt", "w") as f:
        f.write(current_status)

    # Timer & status text
    remaining = int(focus_seconds - (time.time() - start_time))
    mins = remaining // 60
    secs = remaining % 60

    # Status bar on top
    bar_color = (0, 255, 0) if current_status == "FOCUS" else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), bar_color, -1)

    # Big "FOCUS / UNFOCUS" text in bottom-left
    cv2.putText(frame, f"{current_status}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, bar_color, 4)

    # Timer in bottom-right
    timer_text = f"{mins:02}:{secs:02} remaining"
    cv2.putText(frame, timer_text, (frame.shape[1] - 280, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Study Buddy AI", frame)

    if remaining <= 0 or cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()