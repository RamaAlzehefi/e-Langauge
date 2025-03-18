import streamlit as st
import os
os.system("pip install opencv-python-headless")
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer  # إضافة هذه السطر

# باقي الأكواد الأخرى مثل تحميل النموذج MediaPipe و tensorflow

# قم بإضافة هذه الوظيفة بعد تهيئة الكاميرا أو بعد الكود المتعلق بالواجهة
def main():
    st.title("نظام التعرف على لغة الإشارة للصم والبكم")
    
    # استخدام streamlit-webrtc لالتقاط الفيديو
    webrtc_streamer(key="camera")  # إضافة هذا السطر لتشغيل الكاميرا
    
    # باقي الكود مثل تصنيف الإشارة والمعالجة

if __name__ == "__main__":
    main()
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

# كلاس لمعالجة الفيديو باستخدام WebRTC
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # استدعاء دالة تصنيف الإشارة
        gesture, hand_landmarks, confidence = classify_gesture(img)
        
        # رسم معالم اليد إذا تم اكتشافها
        if hand_landmarks:
            for landmarks in hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, landmarks, mp_hands.HAND_CONNECTIONS)
        
        # إضافة النص على الشاشة
        if gesture:
            img = draw_text_with_arabic(img, f"الإشارة: {gesture}", (img.shape[1] // 2, 50), font_size=48)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# الدالة الرئيسية للتطبيق
def main():
    st.title("نظام التعرف على لغة الإشارة للصم والبكم")
    
    # اختيار مصدر الإدخال
    input_source = st.radio("اختر مصدر الإدخال:", ["كاميرا الويب", "تحميل صورة"])
    
    # معالجة الصور المرفوعة
    if input_source == "تحميل صورة":
        uploaded_file = st.file_uploader("اختر صورة من جهازك", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            frame = process_uploaded_image(image_bytes)
            gesture, hand_landmarks, confidence = classify_gesture(frame)
            
            # رسم المعالم إذا تم اكتشاف يد
            if hand_landmarks:
                for landmarks in hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # إضافة النص على الصورة
            if gesture:
                frame = draw_text_with_arabic(frame, f"الإشارة: {gesture}", (frame.shape[1] // 2, 50), font_size=48)
                st.write(f"الإشارة المكتشفة: {gesture}")
                if confidence:
                    st.write(f"نسبة الثقة: {confidence:.2%}")
            else:
                st.write("لم يتم الكشف عن أي إشارة")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="الصورة المعالجة", use_column_width=True)
    
    # تشغيل الكاميرا باستخدام WebRTC
    else:
        st.write("اضغط على زر الإيقاف لإنهاء العرض")
        
        webrtc_streamer(
    key="camera",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        "iceTransportPolicy": "relay",
    },
)


# تشغيل التطبيق
if __name__ == "__main__":
    main()
