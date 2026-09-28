import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="FitBuddy AI", layout="wide")
st.title("FitBuddy - AI Fitness Coach 🏋️")
st.write("Push-up / Squat Counter using MediaPipe")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a,b,c):
    a=np.array(a); b=np.array(b); c=np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle>180.0: angle=360-angle
    return angle

menu = st.sidebar.selectbox("Select Exercise", ["Push-up", "Squat"])
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
count_text = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and run:
            ret, frame = cap.read()
            if not ret: break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                lm = results.pose_landmarks.landmark
                if menu=="Push-up":
                    shoulder=[lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow=[lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist=[lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle=calculate_angle(shoulder, elbow, wrist)
                    if angle>160: stage="up"
                    if angle<70 and stage=="up": stage="down"; counter+=1
                else: # Squat
                    hip=[lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee=[lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle=[lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle=calculate_angle(hip, knee, ankle)
                    if angle>160: stage="up"
                    if angle<90 and stage=="up": stage="down"; counter+=1
                count_text.metric("Count", counter)
            except: pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            FRAME_WINDOW.image(image, channels="BGR")
        cap.release()
else:
    st.info("Check 'Start Camera' to begin. Allow camera permission.")