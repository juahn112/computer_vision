import cv2

# Haar 모델 로딩
front_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# DNN 모델 로딩
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# 꽃 이미지 로딩 (알파 포함)
flower = cv2.imread('flower.png', cv2.IMREAD_UNCHANGED)

# 예외 처리
if front_haar.empty() or profile_haar.empty():
    raise RuntimeError("Haar Cascade 로딩 실패")
if flower is None or flower.shape[2] != 4:
    raise RuntimeError("flower.png 파일이 없거나 알파 채널이 없음")

# 웹캠 연결
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("웹캠 열기 실패")

print("모든 리소스 로딩 완료")

# 노출 수동 조절 UI
cv2.namedWindow('Webcam View')
cv2.createTrackbar('Exposure', 'Webcam View', 50, 100, lambda x: None)

# 초기 노출 설정
prev_exposure_val = -1

while True:
    ret, frame = cam.read()
    if not ret:
        print("캠 프레임 읽기 실패")
        break

    # 슬라이더에서 노출 값 받아오고 변화 있을 때만 설정
    exposure_val = cv2.getTrackbarPos("Exposure", "Webcam View")
    if exposure_val != prev_exposure_val:
        mapped_exposure = max(-6, -10 + (exposure_val / 100) * 9)
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 모드
        cam.set(cv2.CAP_PROP_EXPOSURE, mapped_exposure)
        prev_exposure_val = exposure_val

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    front_faces = front_haar.detectMultiScale(gray, 1.1, 5)
    profile_faces = profile_haar.detectMultiScale(gray, 1.1, 5)

    # 좌측 측면 감지를 위해 이미지 좌우반전 후 탐지
    flipped_gray = cv2.flip(gray, 1)
    flipped_faces = profile_haar.detectMultiScale(flipped_gray, 1.1, 5)

    # 원래 좌표로 변환
    for (x, y, w, h) in flipped_faces:
        x = gray.shape[1] - x - w
        profile_faces = list(profile_faces)
        profile_faces.append((x, y, w, h))

    all_faces = list(front_faces) + list(profile_faces)

    # 얼굴 위에 꽃 이미지 덮기
    for (x, y, w, h) in all_faces:
        face_roi = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (300, 300), (104, 177, 123))
        net.setInput(blob)
        detections = net.forward()
        confidence = detections[0, 0, 0, 2]

        if confidence > 0.5:
            scale = 1.5
            new_w, new_h = int(w * scale), int(h * scale)
            x_offset = max(x - (new_w - w) // 2, 0)
            y_offset = max(y - (new_h - h) // 2, 0)

            flower_resized = cv2.resize(flower, (new_w, new_h))
            flower_rgb = flower_resized[:, :, :3]
            alpha = flower_resized[:, :, 3] / 255.0

            y1, y2 = y_offset, min(y_offset + new_h, frame.shape[0])
            x1, x2 = x_offset, min(x_offset + new_w, frame.shape[1])
            flower_part = flower_rgb[:y2 - y1, :x2 - x1]
            alpha_part = alpha[:y2 - y1, :x2 - x1]

            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                    alpha_part * flower_part[:, :, c] +
                    (1 - alpha_part) * frame[y1:y2, x1:x2, c]
                )

    cv2.imshow('Webcam View', frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
