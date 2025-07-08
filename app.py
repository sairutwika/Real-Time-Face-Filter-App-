# pip install streamlit opencv-python mediapipe qrcode numpy pillow

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import qrcode
from PIL import Image
import datetime
import io

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="🎭 Face Filter Fun", page_icon="🎩", layout="wide")

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #a8edea, #fed6e3);
            color: #000000;
        }
        h1, h2, h3, h4, h5, h6, p, div, span, .element-container {
            color: #000000 !important;
        }
        .stButton>button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #000000;
            border-radius: 10px;
            padding: 8px 20px;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown("<h1 style='text-align:center;'>🎭 Real-Time Face Filter App 🎩</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Load Filters ------------------
mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)
glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
pyrate = cv2.imread("pyrate.png", cv2.IMREAD_UNCHANGED)
crown = cv2.imread("crown.png", cv2.IMREAD_UNCHANGED)

for name, img in [("mustache.png", mustache), ("hat.png", hat), 
                  ("glasses.png", glasses), ("pyrate.png", pyrate),
                  ("crown.png", crown)]:
    if img is None or img.shape[2] != 4:
        st.error(f"'{name}' missing or invalid RGBA image.")
        st.stop()

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("### 🎨 Choose a Filter")
    filter_option = st.radio("Select a filter", [
        "Mustache Only",
        "Hat Only",
        "Mustache & Hat",
        "Glasses Only",
        "Pirate Hat",
        "Grayscale Filter",
        "Crown"
    ])

    st.markdown("---")
    st.subheader("🔗 QR Code Generator")
    qr_input = st.text_input("Enter link to generate QR")
    if qr_input:
        qr = qrcode.make(qr_input)
        st.image(qr, caption="Scan this QR", use_container_width=True)

# ------------------ MediaPipe Setup ------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=5)

# ------------------ Helper ------------------
def overlay_rgba(bg, overlay, x, y, w, h):
    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    b, g, r, a = cv2.split(overlay)
    alpha = a / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    h_bg, w_bg = bg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(x + w, w_bg), min(y + h, h_bg)

    cropped_w = x1 - x0
    cropped_h = y1 - y0
    if cropped_w <= 0 or cropped_h <= 0:
        return bg

    overlay_slice = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    bg_roi = (slice(y0, y1), slice(x0, x1))

    fg = cv2.merge([b, g, r])[overlay_slice]
    alpha_roi = alpha[overlay_slice]
    bg_part = bg[bg_roi]

    blended = cv2.convertScaleAbs(fg * alpha_roi + bg_part * (1 - alpha_roi))
    bg[bg_roi] = blended
    return bg

# ------------------ Live Feed ------------------
frame_placeholder = st.empty()
cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_count = 0
        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)
            for face in results.multi_face_landmarks:
                landmarks = face.landmark

                def to_px(idx):
                    pt = landmarks[idx]
                    return int(pt.x * w), int(pt.y * h)

                x1, y1 = to_px(13)
                x2, y2 = to_px(14)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                x_l, _ = to_px(127)
                x_r, _ = to_px(356)
                face_width = x_r - x_l

                if filter_option == "Mustache Only":
                    m_w, m_h = face_width, int(face_width * 0.3)
                    m_x, m_y = cx - m_w // 2, cy - int(m_h * 0.75)
                    frame = overlay_rgba(frame, mustache, m_x, m_y, m_w, m_h)

                elif filter_option == "Hat Only":
                    x_f, y_f = to_px(10)
                    h_w = int(face_width * 1.6)
                    h_h = int(h_w * 0.9)
                    h_x, h_y = x_f - h_w // 2, y_f - int(h_h * 0.8)
                    frame = overlay_rgba(frame, hat, h_x, h_y, h_w, h_h)

                elif filter_option == "Mustache & Hat":
                    m_w, m_h = face_width, int(face_width * 0.3)
                    m_x, m_y = cx - m_w // 2, cy - int(m_h * 0.75)
                    frame = overlay_rgba(frame, mustache, m_x, m_y, m_w, m_h)

                    x_f, y_f = to_px(10)
                    h_w = int(face_width * 1.6)
                    h_h = int(h_w * 0.9)
                    h_x, h_y = x_f - h_w // 2, y_f - int(h_h * 0.8)
                    frame = overlay_rgba(frame, hat, h_x, h_y, h_w, h_h)

                elif filter_option == "Glasses Only":
                    left_eye, right_eye = to_px(33), to_px(263)
                    g_w = abs(right_eye[0] - left_eye[0]) + 60
                    g_h = int(g_w * 0.4)
                    g_x = left_eye[0] - 30
                    g_y = left_eye[1] - int(g_h / 2)
                    frame = overlay_rgba(frame, glasses, g_x, g_y, g_w, g_h)

                elif filter_option == "Pirate Hat":
                    x_top, y_top = to_px(168)
                    h_w = int(face_width * 1.6)
                    h_h = int(h_w * 0.9)
                    h_x = x_top - h_w // 2
                    h_y = y_top - int(h_h * 1.1)
                    frame = overlay_rgba(frame, pyrate, h_x, h_y, h_w, h_h)

                elif filter_option == "Crown":
                    x_top, y_top = to_px(10)  # top of forehead
                    c_w = int(face_width * 1.5)
                    c_h = int(c_w * 0.7)
                    c_x = x_top - c_w // 2
                    c_y = y_top - int(c_h * 1.1)  # raised to fit above head
                    frame = overlay_rgba(frame, crown, c_x, c_y, c_w, c_h)

        if filter_option == "Grayscale Filter":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        cv2.putText(frame, f"Detected Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        bordered_frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[150, 200, 255])
        frame_placeholder.image(cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()
