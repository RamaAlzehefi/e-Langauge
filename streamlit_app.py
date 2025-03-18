import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import numpy as np
import tensorflow as tf
import av

# تحميل نموذج الذكاء الاصطناعي
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.h5")

model = load_model()

# إعداد Mediapipe لاكتشاف الأيدي
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# تعريف الإشارات المدعومة
sign_language_classes = [
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "عذرًا",
    "", "طعام", "", "", "مرحبًا", "مساعدة", "منزل", "أنا", "أحبك", "", "", "",
    "", "", "لا", "", "", "لو سمحت", "", "", "", "", "شكرًا", "", "", "",
    "", "", "نعم", ""
]

# دالة تحليل المعالم
def process_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

# دالة التصنيف
def classify_gesture(image):
    image_rgb = np.array(image)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        landmarks_array = np.array(process_landmarks(result.multi_hand_landmarks[0])).reshape(1, -1)
        prediction = model.predict(landmarks_array, verbose=0)
        class_id = np.argmax(prediction[0])
        confidence = prediction[0][class_id]
        return sign_language_classes[class_id], confidence
    return None, None

# دالة تشغيل الفيديو باستخدام WebRTC
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gesture, confidence = classify_gesture(img)

        if gesture:
            text = f"الإشارة: {gesture} | الثقة: {confidence:.2%}"
            cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 🚀 **واجهة التطبيق**
st.title("🔵🟢 التعرف على لغة الإشارة للصم والبكم 🟢🔵")

# ✅ **تشغيل الكاميرا باستخدام streamlit-webrtc**
webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)

# 📌 **إضافة زر إيقاف**
if st.button("إيقاف"):
    st.write("📌 تم إيقاف الكاميرا.")
