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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
    
    cv2.imshow('Webcam View', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cam.destroyAllWindows()