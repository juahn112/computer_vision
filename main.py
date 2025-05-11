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
        if confidence > 0.5:
            scale = 1.5
            new_w = int(w * scale)
            new_h = int(h * scale)

            x_offset = max(x - (new_w - w) // 2, 0)
            y_offset = max(y - (new_h - h) // 2, 0)

            flower_resized = cv2.resize(flower, (new_w, new_h))

            flower_rgb = flower_resized[:, :, :3]
            alpha = flower_resized[:, :, 3] / 255.0

            y1, y2 = y_offset, min(y_offset + new_h, frame.shape[0])
            x1, x2 = x_offset, min(x_offset + new_w, frame.shape[1])
            flower_part = flower_rgb[0:y2 - y1, 0:x2 - x1]
            alpha_part = alpha[0:y2 - y1, 0:x2 - x1]

            for c in range(3):  
                frame[y1:y2, x1:x2, c] = (
                    alpha_part * flower_part[:, :, c] +
                    (1 - alpha_part) * frame[y1:y2, x1:x2, c]
                )
    
    cv2.imshow('Webcam View', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cam.destroyAllWindows()