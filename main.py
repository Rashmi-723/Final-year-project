import os
import random
import pygame
from pathlib import Path
import streamlit as st
import av
import cv2
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.app_logo import add_logo
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Set up Streamlit configuration
st.set_page_config(
    page_title="Vibescape",
    page_icon="🎵",
)

page_bg_img = """
<style>
/* Your existing CSS styling */
</style>
"""
add_logo("https://github.com/NebulaTris/vibescape/blob/main/logo.png?raw=true")
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Vibescape 🎉🎶")
st.sidebar.success("Select a page below.")
st.sidebar.text("Developed by Shambhavi")

st.markdown("**Hey there, emotion explorer! Are you ready for a wild ride through the rollercoaster of feelings?** 🎢🎵")
st.markdown("**Welcome to Vibescape, where our snazzy AI meets your wacky emotional world head-on! We've got our virtual goggles on (nope, not really, but it sounds cool** 😎 **) to analyze your emotions using a webcam. And what do we do with all those emotions, you ask? We turn them into the most toe-tapping, heartwarming, and occasionally hilarious music playlists you've ever heard!** 🕺💃")
st.markdown("**You've heard of Spotify, SoundCloud, and YouTube, right? Well, hold onto your hats because Vibescape combines these musical behemoths into one epic entertainment extravaganza! Now you can dive into your favorite streaming services with a twist — they'll be serving up songs based on your mood!** 🎶")
st.markdown("**Feeling like a happy-go-lucky panda today? We've got a playlist for that! Or perhaps you've got the moody blues? No worries, Vibescape has your back. Our AI wizardry detects your vibes and serves up the tunes that match your moment.** 🐼🎉")
st.markdown("**So, get ready for a whirlwind of emotions and music. Vibescape is here to turn your webcam into a mood ring, your screen into a dance floor, and your heart into a DJ booth. What's next? Well, that's entirely up to you and your ever-changing feelings!**")
st.markdown("**So, strap in** 🚀 **, hit that webcam** 📷 **, and let the musical journey begin! Vibescape is your ticket to a rollercoaster of emotions, all set to your favorite tunes.** 🎢🎵")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Paths for audio folders
neutral_folder_path = 'C:/songs/neutral'
sad_folder_path = 'C:/songs/sad'

# Initialize pygame mixer
pygame.mixer.init()

def play_random_song_from_folder(folder_path):
    """Play a random song from the specified folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
    if not files:
        print(f"No audio files found in the folder: {folder_path}")
        return
    song = random.choice(files)
    song_path = os.path.join(folder_path, song)
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()
    print(f"Playing {song}")

# CWD path
HERE = Path(__file__).parent

model = load_model("model.h5")
label = np.load("label.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

if "run" not in st.session_state:
    st.session_state["run"] = ""
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)  
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        
        lst = []
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)
        
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)
        
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)
        
            lst = np.array(lst).reshape(1, -1)
        
            pred = label[np.argmax(model.predict(lst))]
            
            if pred == "Neutral":
                play_random_song_from_folder(neutral_folder_path)
            elif pred == "Sad":
                play_random_song_from_folder(sad_folder_path)
                
            print(pred)
            cv2.putText(frm, pred, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            np.save("emotion.npy", np.array([pred]))
            st.session_state["emotion"] = pred
       
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS) 
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
    
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

webrtc_streamer(
    key="key", 
    desired_playing_state=st.session_state.get("run", "") == "true",
    mode=WebRtcMode.SENDRECV,  
    rtc_configuration=RTC_CONFIGURATION, 
    video_processor_factory=EmotionProcessor, 
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

col1, col2, col6 = st.columns([1, 1, 1])

with col1:
    start_btn = st.button("Start")
with col6:
    stop_btn = st.button("Stop")

if start_btn:
    st.session_state["run"] = "true"
    st.experimental_set_query_params(run="true")

if stop_btn:
    st.session_state["run"] = "false"
    st.experimental_set_query_params(run="false")

# Retrieve query parameters
query_params = st.experimental_get_query_params()
if query_params.get("run", [""])[0] == "false" and st.session_state["emotion"]:
    st.session_state["emotion"] = ""
    st.success("Your current emotion is: " + st.session_state["emotion"])
    st.subheader("Choose your streaming service")

col3, col4, col5 = st.columns(3) 

with col3:
    if st.button("Spotify"):
        st.write("Opening Spotify...")
        switch_page("Spotify")
with col4:
    if st.button("YouTube"):
        st.write("Opening YouTube...")
        switch_page("YouTube")
with col5:
    if st.button("SoundCloud"):
        st.write("Opening SoundCloud...")
        switch_page("SoundCloud")
