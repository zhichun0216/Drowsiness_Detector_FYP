import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import av
from streamlit_webrtc import webrtc_streamer
from streamlit_option_menu import option_menu
import threading
import pygame
import time



model = load_model("drowiness_new6.h5")

labels_new = ["yawn", "no_yawn"]
IMG_SIZE = 145

def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.set_volume(0.1)
    pygame.mixer.music.play()

def prepare(image_array, face_cas_path="haarcascade_frontalface_default.xml"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+face_cas_path)
    faces = face_cascade.detectMultiScale(image_array, 1.3, 3)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_color = img[y:y+h, x:x+w]
        resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
        return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3), [x,y,w,h]
    return 0, []



def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = np.array(img)
    print(time.time())
    try:
        face = [prepare(img)[0]]
        location = prepare(img)[1]
        if np.array(face).any():
            prediction = model.predict(face)
            print(time.time())

            if np.argmax(prediction) == 0:
                str = "Drowsiness Alert!!"
                t = threading.Thread(target=play_audio, args=("alert.mp3",))
                t.start()
            else:
                d_count = 0
                str = "Normal"
            print(str)
        if location != []:
            cv2.putText(img, text= str, org=(location[0],location[1]),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,255),
                    thickness=2, lineType=cv2.LINE_AA)
    except Exception as e:
        print(e)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

with st.sidebar:
    choose = option_menu("Drowsiness Dectector", ["About This App", "Detector"],
                         icons=['app-indicator', 'camera-fill'],
                         menu_icon="car-front-fill", default_index=0,
                         styles={
        "container": {"padding": "5!important"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#aaa"},
        "menu-title" : {"font-size": "17px"}
    }
)

if choose == "About This App":
    st.header("About This App")
    st.write("This app is designed to detect signs of fatigue or drowsiness in a driver and alert them to take a break or rest. This is particularly important in long-distance driving, as fatigue can greatly increase the risk of accidents. This system used convolutional neural network(CNN) with the help of openCV and Dlib to dectect any drowsy.")
    st.header("How to Use")
    st.write("1. Go the the ""Detector"" menu from the side navigation bar")
    st.write("2. Give the access of the camera")
    st.write("3. Click Start")
    st.write("4. The system will try to search for your face and detect your drowsiness status")

if choose == "Detector":
        webrtc_streamer(
        key="example",
        video_frame_callback=callback,
        rtc_configuration={  # Add this line
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
