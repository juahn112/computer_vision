import cv2
import cv2.data

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("face cascade error")
else:
    print("success")

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("cam is not opened. error")
    exit()
else:
    print("cam is opened. success")
while True:
    ret, frame = cam.read()
    if not ret:
        print("ret is False")
        break
    
    cv2.imshow('Webcam View', frame)

    cv2.waitKey(1)