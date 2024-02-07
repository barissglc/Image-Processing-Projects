import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpYuzMesh = mp.solutions.face_mesh
YuzMesh = mpYuzMesh.FaceMesh(max_num_faces = 2)
mp.solutions.drawing_utils = mp.solutions.drawing_utils
drawSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1) # çizgi kalınlığı ve çember yarıçapı
#burada aslinda çizim özelliklerini degistiriyoruz

previousTime = 0
currentTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = YuzMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, faceLms, mpYuzMesh.FACEMESH_CONTOURS,
            drawSpec, drawSpec)

        for id, lm in enumerate(faceLms.landmark):
            h,w,_ = img.shape
            cx,cy = int(lm.x*w), int(lm.y*h)
            print(id, cx, cy)


    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)








    cv2.imshow("img", img)
    cv2.waitKey(50)



