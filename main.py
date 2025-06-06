import cv2

front_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
flower = cv2.imread('flower.png', cv2.IMREAD_UNCHANGED)

if front_haar.empty() or profile_haar.empty():
    raise RuntimeError("Haar cascade load error")
print("Cascade load success")

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Camera not opened")
print("Camera opened")

def nothing(x):
    pass

cv2.namedWindow('Webcam View')
cv2.createTrackbar('Exposure', 'Webcam View', 10, 100, nothing)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    
    exposure_val = cv2.getTrackbarPos("Exposure", "Webcam View")
    mapped_exposure = max(-6, -10 + (exposure_val / 100) * 9)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_EXPOSURE, mapped_exposure)

    
    if frame.mean() < 10:
        cv2.putText(frame, "Too dark for detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Webcam View', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    front_faces = front_haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    profile_faces = profile_haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    flipped_faces = profile_haar.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=5)

    all_faces = list(front_faces) + list(profile_faces)
    for (x, y, w, h) in flipped_faces:
        real_x = gray.shape[1] - x - w
        all_faces.append((real_x, y, w, h))

    for (x, y, w, h) in all_faces:
        face_roi = frame[y:y + h, x:x + w]
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
cv2.destroyAllWindows()
