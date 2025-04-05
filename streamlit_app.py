import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import tflite_runtime.interpreter as tflite

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="sign_language_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model()

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Arabic labels
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        landmarks = process_landmarks(result.multi_hand_landmarks[0])
        if len(result.multi_hand_landmarks) > 1:
            landmarks += process_landmarks(result.multi_hand_landmarks[1])
        else:
            landmarks += pad_landmarks()

        input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        class_id = np.argmax(output_data)
        confidence = output_data[class_id]
        return sign_language_classes[class_id], result.multi_hand_landmarks, confidence
    return None, None, None

def draw_text_with_arabic(frame, text, position, font_size=48, color=(0, 255, 0)):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    pos = (position[0] - (bbox[2] - bbox[0]) // 2, position[1] - (bbox[3] - bbox[1]) // 2)
    draw.text(pos, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def process_uploaded_image(image_bytes):
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

def main():
    st.title("نظام التعرف على لغة الإشارة للصم والبكم")

    source = st.radio("اختر مصدر الإدخال:", ["تحميل صورة"])

    if source == "تحميل صورة":
        file = st.file_uploader("اختر صورة", type=['jpg', 'jpeg', 'png'])
        if file:
            img = process_uploaded_image(file.read())
            gesture, landmarks, conf = classify_gesture(img)
            if landmarks:
                for lm in landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)
            if gesture:
                img = draw_text_with_arabic(img, f"الإشارة: {gesture}", (img.shape[1] // 2, 50))
                st.write(f"الإشارة المكتشفة: {gesture}")
                st.write(f"نسبة الثقة: {conf:.2%}")
            else:
                st.warning("لم يتم الكشف عن أي إشارة")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="الصورة المعالجة", use_column_width=True)

if __name__ == "__main__":
    main()
