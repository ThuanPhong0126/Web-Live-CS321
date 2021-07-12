import cv2
import numpy as np
import streamlit as st
import dlib
from math import hypot
import time

st.title("Webcam Live")
run = st.checkbox("Run")
frame_window = st.image([])
camera = cv2.VideoCapture(0)
genre = st.sidebar.radio("What's your favorite effect?",
                         ('Normal', 'Scan', 'Pig Nose', 'Blue Eyes', 'Lip'))
if genre == 'Scan':
    scan_option = st.selectbox("Option scan effect:", ('Top to Bottom', 'Left to Right'))

img = np.zeros((480, 640, 3), np.uint8)
i = 0

nose_image = cv2.imread("pig_nose.png")
nose_image = cv2.cvtColor(nose_image, cv2.COLOR_BGR2RGB)
nose_mask = np.zeros((480, 640), np.uint8)

blue_eye_left_image = cv2.imread("eye_left.png")
blue_eye_left_image = cv2.cvtColor(blue_eye_left_image, cv2.COLOR_BGR2RGB)
blue_eye_mask_left = np.zeros((480, 640), np.uint8)

blue_eye_right_image = cv2.imread("eye_right.png")
blue_eye_right_image = cv2.cvtColor(blue_eye_right_image, cv2.COLOR_BGR2RGB)
blue_eye_mask_right = np.zeros((480, 640), np.uint8)

lip_image = cv2.imread('lip.png')
lip_image = cv2.cvtColor(lip_image, cv2.COLOR_BGR2RGB)
lip_mask = np.zeros((480, 640), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if genre == 'Scan':

        frame = np.fliplr(frame)
        h, w = frame.shape[:2]

        if scan_option == 'Left to Right':
            # chạy từ left to right
            img[:, i+1: w, :] = frame[:, i+1: w, :]
            cv2.line(img, (i+1, 0), (i+1, h), (0, 255, 0), 2)
            img[:, i:i+1, :] = frame[:, i:i+1, :]

        else:# chạy từ trên xuống
            img[i + 1:h, :, :] = frame[i + 1:h, :, :]
            cv2.line(img, (0, i + 1), (w, i + 1), (0, 255, 0), 2)
            img[i:i + 1, :, :] = frame[i:i + 1, :, :]

        i += 1

    elif genre == 'Pig Nose':
        nose_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = detector(frame)

        for face in faces:
            #print(face)
            landmarks = predictor(gray_frame, face)

            # Nose coordinates
            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            nose_width = int(hypot(left_nose[0] - right_nose[0],
                                   left_nose[1] - right_nose[1]) * 1.7)
            nose_height = int(nose_width * 0.77)

            # New nose position
            top_left = (int(center_nose[0] - nose_width / 2),
                        int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                            int(center_nose[1] + nose_height / 2))

            # Adding the new nose
            nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
            nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

            nose_area = frame[top_left[1]: top_left[1] + nose_height,
                        top_left[0]: top_left[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)

            frame[top_left[1]: top_left[1] + nose_height,
            top_left[0]: top_left[0] + nose_width] = final_nose
        img = frame
    elif genre == 'Lip':
        lip_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(frame)
        for face in faces:
            # print(face)
            landmarks = predictor(gray_frame, face)

            width = int(abs(landmarks.part(48).x - landmarks.part(54).x)*1.25)
            height = int(width * 0.5)

            top_left = (int(landmarks.part(66).x - width/2),
                         int(landmarks.part(66).y - height/2))
            bottom_right = (int(landmarks.part(66).x + width/2),
                             int(landmarks.part(66).y + height/2))

            lip = cv2.resize(lip_image, (width, height))
            lip_gray = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
            _, lip_mask = cv2.threshold(lip_gray, 25, 255, cv2.THRESH_BINARY_INV)

            lip_area = frame[top_left[1]: top_left[1] + height,
                             top_left[0]: top_left[0] + width]
            lip_area_no_lip = cv2.bitwise_and(lip_area, lip_area, mask=lip_mask)
            final_lip = cv2.add(lip_area_no_lip, lip)

            frame[top_left[1]: top_left[1] + height,
                top_left[0]: top_left[0] + width] = final_lip
        img = frame

    elif genre == "Blue Eyes":
        blue_eye_mask_left.fill(0)
        blue_eye_mask_right.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(frame)
        for face in faces:
            # print(face)
            landmarks = predictor(gray_frame, face)

            # eye coordinates
            width = int(2*abs(landmarks.part(40).x - landmarks.part(36).x))
            height = int(width*1.5)

            # left eye position
            top_left_left = (int(landmarks.part(37).x - width),
                        int(landmarks.part(37).y - height))
            bottom_left_right = (int(landmarks.part(37).x + width),
                            int(landmarks.part(37).y + height))

            # right eye position
            top_right_left = (int(landmarks.part(44).x - width),
                             int(landmarks.part(44).y - height))
            bottom_right_right = (int(landmarks.part(44).x + width),
                                 int(landmarks.part(44).y + height))

            # Adding the new left eye
            blue_eye_left = cv2.resize(blue_eye_left_image, (2*width, 2*height))
            blue_eye_left_gray = cv2.cvtColor(blue_eye_left, cv2.COLOR_BGR2GRAY)
            _, blue_eye_mask_left = cv2.threshold(blue_eye_left_gray, 25, 255, cv2.THRESH_BINARY_INV)

            blue_eye_area_left = frame[top_left_left[1]: top_left_left[1] + height*2,
                        top_left_left[0]: top_left_left[0] + width*2]
            blue_eye_area_no_eye_left = cv2.bitwise_and(blue_eye_area_left, blue_eye_area_left, mask=blue_eye_mask_left)
            final_blue_eye_left = cv2.add(blue_eye_area_no_eye_left, blue_eye_left)

            # Adding the new right eye
            blue_eye_right = cv2.resize(blue_eye_right_image, (2 * width, 2 * height))
            blue_eye_right_gray = cv2.cvtColor(blue_eye_right, cv2.COLOR_BGR2GRAY)
            _, blue_eye_mask_right = cv2.threshold(blue_eye_right_gray, 25, 255, cv2.THRESH_BINARY_INV)

            blue_eye_area_right = frame[top_right_left[1]: top_right_left[1] + height * 2,
                                 top_right_left[0]: top_right_left[0] + width * 2]
            blue_eye_area_no_eye_right = cv2.bitwise_and(blue_eye_area_right, blue_eye_area_right, mask=blue_eye_mask_right)
            final_blue_eye_right = cv2.add(blue_eye_area_no_eye_right, blue_eye_right)

            frame[top_left_left[1]: top_left_left[1] + height*2, top_left_left[0]: top_left_left[0] + width*2] = final_blue_eye_left
            frame[top_right_left[1]: top_right_left[1] + height*2, top_right_left[0]: top_right_left[0] + width*2] = final_blue_eye_right


        img = frame
    else:
        img = frame

    frame_window.image(img)
else:
    st.write('Stopped')