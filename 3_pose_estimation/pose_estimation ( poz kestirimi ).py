import cv2
import mediapipe as mp
import time

mpPoz = mp.solutions.pose
poz = mpPoz.Pose()

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
cap = cv2.VideoCapture("video4.mp4")
#cap = cv2.VideoCapture(0)

while True:
    success, kamera = cap.read()
    imgRGB = cv2.cvtColor(kamera,cv2.COLOR_BGR2RGB)
    results = poz.process(imgRGB)
    print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(kamera, results.pose_landmarks,mpPoz.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w, _ = kamera.shape
            cx,cy = int(lm.x*w), int(lm.y*h)

            if id == 4:
                cv2.circle(kamera, (cx,cy), 9, (255,0,0), cv2.FILLED)


    cTime = time.time()# şuanki zamanı alıyoruz
    fps = 1/(cTime-pTime) # fps hesaplıyoruz
    pTime = cTime # şuanki zamanı geçmiş zamana eşitliyoruz

    cv2.putText(kamera,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)


    kamera = cv2.resize(kamera, (1000, 1000))
    cv2.imshow("kamera", kamera)

    cv2.waitKey(25)