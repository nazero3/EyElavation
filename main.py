import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
import pyautogui
import cv2 as cv
from stuff import *
cap = cv2.VideoCapture(0)

screen_width = pyautogui.size().width
screen_height = pyautogui.size().height
kernel = np.ones((2, 2), np.uint8)

while True:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        faceROI = frame_gray[y:y + h, x:x + w]
        roi = frame[y:y + h, x:x + w]
        # -- In each face, detect eyes
        eyes = eye_cascade.detectMultiScale(faceROI)
        eyes = getleftmosteye(eyes)
        for (x2, y2, w2, h2) in eyes:
            eye = roi[y2:y2 + h2, x2:x2 + w2]
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            bilateral_filter = cv2.bilateralFilter(gray, 3, 25, 25)
            # blur = cv.medianBlur(bilateral_filter, 5)
            # cv.imshow('blur', blur)
            th3 = cv.adaptiveThreshold(bilateral_filter, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY_INV, 51, 10)
            cv.imshow('th3', th3)
            s = sobel(th3)
            cv.imshow('s', s)
            circles = cv2.HoughCircles(s, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=30)
            coe_x = x + x2 + w2 // 2
            coe_y = y + y2 + h2 // 2
            eye_radius = int(round((w2 + h2) * 0.25))
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                circle = min_circle(circles)
                x3, y3, r = circle
                coi_x = x + x2 + x3
                coi_y = y + y2 + y3
                scale_x = calc_scale(screen_width, w2)
                scale_y = calc_scale(screen_height, h2)
                pog_x = calc_pog(screen_width, scale_x, coi_x, coe_x)
                pog_y = calc_pog(screen_height, scale_y, coi_y, coe_y)
                pog_x, pog_y = adjust(pog_x, pog_y)
                pyautogui.moveTo(pog_x, pog_y, duration=0.1)
                cv2.circle(frame, (coi_x, coi_y), r, (36, 255, 12), 1)
                cv2.circle(frame, (coe_x, coe_y), eye_radius, (43, 12, 243), 1)
                break
        break
        e = 0
    cv.imshow('Capture - Face detection', frame)
    if cv.waitKey(10) == 27:
        break
