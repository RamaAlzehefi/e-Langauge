import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.h5")

model = load_model()

sign_language_classes = [
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "عذرًا",
    "", "طعام", "", "", "مرحبًا", "مساعدة", "منزل", "أنا", "أحبك", "", "", "",
    "", "", "لا", "", "", "لو سمحت", "", "", "", "", "شكرًا", "", "", "",
    "", "", "نعم", ""
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

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
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    pos = (position[0] - (bbox[2] - bbox[0]) // 2, position[1] - (bbox[3] - bbox[1]) // 2)
    draw.text(pos, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class SignLanguageProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gesture, hand_landmarks, confidence = classify_gesture(img)
        if hand_landmarks:
            for hand in hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        if gesture:
            img = draw_text_with_arabic(img, f"الإشارة: {gesture}", (img.shape[1] // 2, 50))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("🖐️ Arabic Sign Language Detection – e-Language")
    st.markdown("This app uses your **webcam** to detect **Arabic sign language gestures** live.")

    webrtc_streamer(
        key="sign-lang",
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False}
    )

if __name__ == "__main__":
    main()
