import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.h5")

model = load_model()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Arabic labels for the signs
sign_language_classes = [
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "عذرًا",
    "", "طعام", "", "", "مرحبًا", "مساعدة", "منزل", "أنا", "أحبك", "", "", "",
    "", "", "لا", "", "", "لو سمحت", "", "", "", "", "شكرًا", "", "", "",
    "", "", "نعم", ""
]

def process_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

def pad_landmarks():
    return [0.0] * 63

def classify_gesture(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        landmarks = process_landmarks(result.multi_hand_landmarks[0])
        if len(result.multi_hand_landmarks) > 1:
            landmarks += process_landmarks(result.multi_hand_landmarks[1])
        else:
            landmarks += pad_landmarks()
        prediction = model.predict(np.array(landmarks).reshape(1, -1), verbose=0)
        class_id = np.argmax(prediction[0])
        confidence = prediction[0][class_id]
        return sign_language_classes[class_id], result.multi_hand_landmarks, confidence
    return None, None, None

def draw_text_with_arabic(frame, text, position, font_path="arial.ttf", font_size=48, color=(0, 255, 0)):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = (position[0] - text_width // 2, position[1] - text_height // 2)
    draw.text(position, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def process_uploaded_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def main():
    st.title("📸 Arabic Sign Language Detection")
    mode = st.radio("Choose Input Mode:", ["Webcam", "Image Upload"])

    if mode == "Webcam":
        st.write("Press stop to end the webcam.")
        stop_btn = st.button("Stop Camera")
        placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            gesture, hand_landmarks, confidence = classify_gesture(frame)
            if hand_landmarks:
                for hand in hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if gesture:
                frame = draw_text_with_arabic(frame, f"الإشارة: {gesture}", (frame.shape[1]//3, 50))
                st.write(f"Detected: {gesture} ({confidence:.2%})")
            placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

    else:
        uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            gesture, hand_landmarks, confidence = classify_gesture(frame)
            if hand_landmarks:
                for hand in hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if gesture:
                frame = draw_text_with_arabic(frame, f"الإشارة: {gesture}", (frame.shape[1]//3, 50))
                st.write(f"Detected: {gesture} ({confidence:.2%})")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

if __name__ == "__main__":
    main()
