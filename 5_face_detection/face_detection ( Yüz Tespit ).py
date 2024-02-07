import cv2
import mediapipe as mp

cap = cv2.VideoCapture("video2.mp4")

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection() # burada takip parametresi var mesela eğer 0.1 yazarsak
# yüzü takip ederken daha hassas olur ama yavaş çalışır. 0.9 yazarsak daha hızlı çalışır ama yüzü takip etme hassasiyeti düşer.
# kendi projene göre tune etmen gerekiyor.

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    # print(results.detections)

    if results.detections:
        for id,detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box

            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)
            cv2.rectangle(img, bbox, (0,255,255),2)
            #write human label
            cv2.putText(img, f'Human', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)



    cv2.imshow("img",img)
    cv2.waitKey(10)
