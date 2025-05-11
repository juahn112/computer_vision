import cv2

haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

net = cv2.dnn.readNetFromCaffe('deploy.prototxt','res10_300x300_ssd_iter_140000.caffemodel')

flower = cv2.imread('flower.png', cv2.IMREAD_UNCHANGED)

if haar.empty():
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

    haar_faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in haar_faces:
        face_roi = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()

        confidence = detections[0, 0, 0, 2]  
        if confidence > 0.6:
            flower_resized = cv2.resize(flower, (w,h))

            flower_rgb = flower_resized[:, :, :3]
            alpha_mask = flower_resized[:,:,3] / 255.0

            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha_mask * flower_rgb[:,:,c]+
                    (1-alpha_mask) * frame[y:y+h, x:x+w, c]
                )
    
    cv2.imshow('Webcam View', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cam.destroyAllWindows()