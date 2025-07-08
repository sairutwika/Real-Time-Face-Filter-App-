# Project: Face Filter with Up to 5 Faces
# pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np

# File paths (update if needed)
mustache_path = r"C:\Users\sairu\OneDrive\Desktop\class1\mustache.png"
hat_path = r"C:\Users\sairu\OneDrive\Desktop\class1\hat.png"

# Load images
mustache_png = cv2.imread(mustache_path, cv2.IMREAD_UNCHANGED)
hat_png = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)

# Check image validity
for name, img in (("mustache.png", mustache_png), ("hat.png", hat_png)):
    if img is None or img.shape[2] < 4:
        raise FileNotFoundError(f"Invalid file: {name} (must be PNG with alpha channel)")

# MediaPipe FaceMesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# RGBA overlay function
def overlay_rgba(background, overlay, x, y, w, h):
    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    b, g, r, a = cv2.split(overlay)
    alpha = a.astype(float) / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    h_bg, w_bg = background.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(x + w, w_bg), min(y + h, h_bg)

    overlay_slice = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    background_roi = (slice(y0, y1), slice(x0, x1))

    foreground = cv2.merge([b, g, r])[overlay_slice]
    alpha_roi = alpha[overlay_slice]
    bg_roi = background[background_roi]

    blended = cv2.convertScaleAbs(foreground * alpha_roi + bg_roi * (1 - alpha_roi))
    background[background_roi] = blended
    return background

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# Main loop
while True:
    ok, frame = cap.read()
    if not ok:
        print("Empty frame received")
        break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            def to_px(idx):
                pt = landmarks[idx]
                return int(pt.x * w_frame), int(pt.y * h_frame)

            # Mustache coordinates
            lib_x1, lib_y1 = to_px(13)
            lib_x2, lib_y2 = to_px(14)
            lib_x = (lib_x1 + lib_x2) // 2
            lib_y = (lib_y1 + lib_y2) // 2

            # Hat coordinates
            left_temple_x, _ = to_px(127)
            right_temple_x, _ = to_px(356)
            forehead_x, forehead_y = to_px(10)
            face_w = right_temple_x - left_temple_x

            # Add mustache
            must_w = face_w
            must_h = int(must_w * 0.3)
            must_x = lib_x - must_w // 2
            must_y = lib_y - int(must_h * 0.75)
            frame = overlay_rgba(frame, mustache_png, must_x, must_y, must_w, must_h)

            # Add hat
            hat_w = int(face_w * 1.6)
            hat_h = int(hat_w * 0.9)
            hat_x = forehead_x - hat_w // 2
            hat_y = forehead_y - int(hat_h * 0.8)
            frame = overlay_rgba(frame, hat_png, hat_x, hat_y, hat_w, hat_h)

    # Show result
    cv2.imshow("🧑 Face Filter (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
