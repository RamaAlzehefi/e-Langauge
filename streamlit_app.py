import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# Load the trained TensorFlow model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.h5")

model = load_model()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Arabic sign language class labels
sign_language_classes = [
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "عذرًا",
    "", "طعام", "", "", "مرحبًا", "مساعدة", "منزل", "أنا", "أحبك", "", "", "",
    "", "", "لا", "", "", "لو سمحت", "", "", "", "", "شكرًا", "", "", "",
    "", "", "نعم", ""
]

# Helper: convert landmarks into flat array
def process_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

# Add padding if second hand not present
def pad_landmarks():
    return [0.0] * 63

# Prediction function
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

# Draw Arabic text properly on the frame
def draw_text_with_arabic(frame, text, position, font_path="arial.ttf", font_size=48, color=(0, 255, 0)):
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    position = (position[0] - (bbox[2] - bbox[0]) // 2, position[1] - (bbox[3] - bbox[1]) // 2)
    draw.text(position, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Main App (Webcam Only)
def main():
    st.title("📸 Arabic Sign Language Detection – e-Language")
    st.write("اضغط على زر الإيقاف لإنهاء العرض")

    stop_btn = st.button("إيقاف")
    placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.error("فشل في تشغيل كاميرا الويب.")
            break

        gesture, landmarks, confidence = classify_gesture(frame)
        if landmarks:
            for lm in landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        if gesture:
            frame = draw_text_with_arabic(frame, f"الإشارة: {gesture}", (frame.shape[1] // 2, 50))
            st.write(f"الإشارة المكتشفة: {gesture} ({confidence:.2%})")
        placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()
