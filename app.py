import os
import random
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import subprocess  # For launching external applications

# Initialize MediaPipe
holistic = mp.solutions.holistic
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Load model and labels
model = load_model("model.h5")
label = np.load("label.npy")

# Define paths for audio folders
emotion_folders = {
    "Angry": r'C:\music\Angry',
    "Sad": r'C:\music\Sad',
    "Happy": r'C:\music\Happy', 
    "Surprise":r'C:\music\Surprise',
    "Fear":r'C:\music\Fear'# Fixed the folder path
}

def play_random_song_from_folder(folder_path):
    """Play a random song from the specified folder using an external music player."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
    if not files:
        print(f"No audio files found in the folder: {folder_path}")
        return
    
    song = random.choice(files)
    song_path = os.path.join(folder_path, song)
    
    # Launch the default music player with the audio file
    try:
        if os.name == 'nt':  # For Windows
            os.startfile(song_path)
        elif os.name == 'posix':  # For macOS or Linux
            subprocess.call(['open' if sys.platform == 'darwin' else 'xdg-open', song_path])
        print(f"Playing: {song_path}")
    except Exception as e:
        print(f"Failed to play song: {e}")

def detect_emotion(frame):
    """Detect emotion from the given frame."""
    frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = holis.process(frm)

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
        return pred
    return None

def main():
    """Main function to run the video capture and emotion detection."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    
    if not cap.isOpened():
        print("Error: Camera not found or cannot be opened.")
        return

    # Set the resolution (width, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Detect emotion
        emotion = detect_emotion(frame)
        
        # Display emotion prediction on the frame
        if emotion:
            # Draw the text on the frame
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Play corresponding audio
            if emotion in emotion_folders:
                play_random_song_from_folder(emotion_folders[emotion])
        
        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
