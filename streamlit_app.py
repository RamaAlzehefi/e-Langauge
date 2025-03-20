import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

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

# Function to draw Arabic text using Pillow
def draw_text_with_arabic(frame, text, position, font_path="arial.ttf", font_size=48, color=(0, 255, 0)):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate text dimensions
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    # Adjust position
    position = (position[0] - text_width // 2, position[1] - text_height // 2)
    
    # Draw text
    draw.text(position, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# Function to process uploaded image
def process_uploaded_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

# Class for Video Processing in WebRTC
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert to numpy array
        gesture, hand_landmarks, confidence = classify_gesture(img)

        if hand_landmarks:
            for landmarks in hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, landmarks, mp_hands.HAND_CONNECTIONS
                )

        if gesture:
            text = f"الإشارة: {gesture} | الثقة: {confidence:.2%}"
            cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")  # Return modified frame

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

        webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False}
        )

if __name__ == "__main__":
    main()
