import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# تحميل نموذج تعلم الآلة
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.h5")

model = load_model()

# إعداد MediaPipe لاكتشاف اليد
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# قائمة التصنيفات للإشارات
sign_language_classes = [
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "عذرًا",
    "", "طعام", "", "", "مرحبًا", "مساعدة", "منزل", "أنا", "أحبك", "", "", "",
    "", "", "لا", "", "", "لو سمحت", "", "", "", "", "شكرًا", "", "", "",
    "", "", "نعم", ""
]

# دالة معالجة المعالم اليدوية
def process_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

# دالة تعبئة بيانات اليد في حال عدم توفر يد ثانية
def pad_landmarks():
    return [0.0] * 63

# دالة تصنيف الإشارة
def classify_gesture(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        combined_landmarks = []
        
        # معالجة اليد الأولى
        combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[0]))
        
        # معالجة اليد الثانية إذا كانت موجودة، أو إضافة بيانات فارغة
        if len(result.multi_hand_landmarks) > 1:
            combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[1]))
        else:
            combined_landmarks.extend(pad_landmarks())
            
        # تحويل البيانات إلى مصفوفة وتمريرها للنموذج
        landmarks_array = np.array(combined_landmarks).reshape(1, -1)
        prediction = model.predict(landmarks_array, verbose=0)
        class_id = np.argmax(prediction[0])
        confidence = prediction[0][class_id]
        
        return sign_language_classes[class_id], result.multi_hand_landmarks, confidence
    
    return None, None, None

# دالة رسم النص العربي على الصورة
def draw_text_with_arabic(frame, text, position, font_path="arial.ttf", font_size=48, color=(0, 255, 0)):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    # تحميل الخط العربي
    font = ImageFont.truetype(font_path, font_size)
    
    # حساب حجم النص ووضعه في منتصف المكان المطلوب
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    position = (position[0] - text_width // 2, position[1] - text_height // 2)
    
    # رسم النص على الصورة
    draw.text(position, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# دالة لمعالجة الصور المرفوعة
def process_uploaded_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

# كلاس لمعالجة الفيديو باستخدام WebRTC
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # استدعاء دالة تصنيف الإشارة
        gesture, hand_landmarks, confidence = classify_gesture(img)
        
        # رسم معالم اليد إذا تم اكتشافها
        if hand_landmarks
