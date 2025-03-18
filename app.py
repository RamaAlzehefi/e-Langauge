import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw

# Initialize session state
if 'stop_button' not in st.session_state:
    st.session_state['stop_button'] = False
if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = True

# Cache model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.h5")

model = load_model()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Define class names
sign_language_classes = [
    "", "", "", "", "", "", "", "", "", "", "", "", "", "", "عذرًا",
    "", "طعام", "", "", "مرحبًا", "مساعدة", "منزل", "أنا", "أحبك", "", "", "",
    "", "", "لا", "", "", "لو سمحت", "", "", "", "", "شكرًا", "", "", "",
    "", "", "نعم", ""
]

# Function to process landmarks
def process_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

# Function to pad landmarks
def pad_landmarks():
    return [0.0] * 63

# Function to classify gestures
def classify_gesture(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        combined_landmarks = []
        
        # Process first hand
        combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[0]))
        
        # Process second hand or pad
        if len(result.multi_hand_landmarks) > 1:
            combined_landmarks.extend(process_landmarks(result.multi_hand_landmarks[1]))
        else:
            combined_landmarks.extend(pad_landmarks())
            
        # Make prediction
        landmarks_array = np.array(combined_landmarks).reshape(1, -1)
        prediction = model.predict(landmarks_array, verbose=0)
        class_id = np.argmax(prediction[0])
        confidence = prediction[0][class_id]
        
        return sign_language_classes[class_id], result.multi_hand_landmarks, confidence
    
    return None, None, None

# Function to draw Arabic text using Pillow with enhanced formatting
def draw_text_with_arabic(frame, text, position, font_path="arial.ttf", font_size=48, color=(0, 255, 0)):
    # Convert the frame to a Pillow image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    # Load a font that supports Arabic
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate the text dimensions using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    # Adjust the position to center the text
    position = (position[0] - text_width // 2, position[1] - text_height // 2)
    
    # Draw the Arabic text
    draw.text(position, text, font=font, fill=color)
    
    # Convert the image back to OpenCV format
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# Function to process uploaded image
def process_uploaded_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

# Main function
def main():
    st.title("نظام التعرف على لغة الإشارة للصم والبكم")
    
    # Input source selection
    input_source = st.radio("اختر مصدر الإدخال:", ["كاميرا الويب", "تحميل صورة"])
    
    if input_source == "تحميل صورة":
        uploaded_file = st.file_uploader("اختر صورة من جهازك", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            frame = process_uploaded_image(image_bytes)
            gesture, hand_landmarks, confidence = classify_gesture(frame)
            
            if hand_landmarks:
                for landmarks in hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            if gesture:
                frame = draw_text_with_arabic(frame, f"الإشارة: {gesture}", (frame.shape[1] // 2, 50), font_size=48)
                st.write(f"الإشارة المكتشفة: {gesture}")
                if confidence:
                    st.write(f"نسبة الثقة: {confidence:.2%}")
            else:
                st.write("لم يتم الكشف عن أي إشارة")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="الصورة المعالجة", use_column_width=True)
    
    else:  # Webcam
        st.write("اضغط على زر الإيقاف لإنهاء العرض")
        video_placeholder = st.empty()
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        stop_button = st.button("إيقاف")
        
        cap = cv2.VideoCapture(0)
        
        try:
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("فشل في تشغيل كاميرا الويب")
                    break

                gesture, hand_landmarks, confidence = classify_gesture(frame)

                if hand_landmarks:
                    for landmarks in hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, landmarks, mp_hands.HAND_CONNECTIONS)

                if gesture:
                    frame = draw_text_with_arabic(frame, f"الإشارة: {gesture}", (frame.shape[1] // 2, 50), font_size=48)
                    prediction_placeholder.text(f"الإشارة المكتشفة: {gesture}")
                    if confidence:
                        confidence_placeholder.write(f"نسبة الثقة: {confidence:.2%}")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        finally:
            cap.release()
            st.session_state['camera_running'] = False

if __name__ == "__main__":
    main()
