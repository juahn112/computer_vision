import cv2

# Haar cascade classifiers
front_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# DNN 모델 로드
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# 오버레이 이미지 로드
flower = cv2.imread('flower.png', cv2.IMREAD_UNCHANGED)
man = cv2.imread('man.png', cv2.IMREAD_UNCHANGED)

if front_haar.empty() or profile_haar.empty():
    raise RuntimeError("하르 캐스캐이드 로드 실패")
if net.empty():
    raise RuntimeError("DNN 모델 로드 실패")
if flower is None or man is None:
    raise RuntimeError("오버레이 이미지 로드 실패")

# 카메라 열기
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("웹캠 열기 실패")

# 트랙바 설정
def nothing(x): pass

cv2.namedWindow('Webcam View')
cv2.createTrackbar('Exposure', 'Webcam View', 50, 100, nothing)  # 노출
cv2.createTrackbar('Overlay Type', 'Webcam View', 0, 1, nothing)  # 0: 꽃, 1: 남자

while True:
    ret, frame = cam.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # 노출 조절
    exposure_val = cv2.getTrackbarPos("Exposure", "Webcam View")
    mapped_exposure = max(-6, -10 + (exposure_val / 100) * 9)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_EXPOSURE, mapped_exposure)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    front_faces = front_haar.detectMultiScale(gray, 1.1, 5)
    profile_faces = profile_haar.detectMultiScale(gray, 1.1, 5)
    all_faces = list(front_faces) + list(profile_faces)

    flipped_gray = cv2.flip(gray, 1)
    flipped_faces = profile_haar.detectMultiScale(flipped_gray, 1.1, 5)

    for (x, y, w, h) in flipped_faces:
        real_x = gray.shape[1] - x - w
        all_faces.append((real_x, y, w, h))

    overlay_choice = cv2.getTrackbarPos("Overlay Type", "Webcam View")
    overlay_img = flower if overlay_choice == 0 else man

    for (x, y, w, h) in all_faces:
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

            overlay_resized = cv2.resize(overlay_img, (new_w, new_h))
            overlay_rgb = overlay_resized[:, :, :3]
            alpha = overlay_resized[:, :, 3] / 255.0

            y1, y2 = y_offset, min(y_offset + new_h, frame.shape[0])
            x1, x2 = x_offset, min(x_offset + new_w, frame.shape[1])
            overlay_part = overlay_rgb[0:y2 - y1, 0:x2 - x1]
            alpha_part = alpha[0:y2 - y1, 0:x2 - x1]

            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha_part * overlay_part[:, :, c] +
                    (1 - alpha_part) * frame[y1:y2, x1:x2, c]
                )

    cv2.imshow('Webcam View', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
