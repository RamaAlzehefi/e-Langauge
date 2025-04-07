import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display
import cv2

# Load your trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Labels
labels = [
    'السلام عليكم', 'شكرا', 'مع السلامة', 'نعم', 'لا',
    'صباح الخير', 'مساء الخير', 'كيف حالك؟', 'بخير', 'تمام'
]

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.title("📷 تطبيق ترجمة لغة الإشارة")
st.markdown("هذا التطبيق يستخدم الكاميرا للتعرف على إشارات لغة الإشارة العربية.")

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            if len(landmarks) == 42:
                prediction = model.predict(np.array([landmarks]))
                predicted_label = labels[np.argmax(prediction)]
                reshaped_text = arabic_reshaper.reshape(predicted_label)
                bidi_text = get_display(reshaped_text)
                cv2.putText(image, bidi_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image

webrtc_streamer(key="sign-lang", video_processor_factory=VideoProcessor)

st.warning("تأكد من السماح للمتصفح باستخدام الكاميرا.")
