import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0) # kamerayi sectik
cap.set(3,640)
cap.set(4,480) #burada kamera boyutlarini ayarladik. zaten default halde bu halde

El = mp.solutions.hands
Eller = El.Hands()
Cizim = mp.solutions.drawing_utils

tipIds = [4,8,12,16,20] # bu parmak noktalarinin idleri



while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = Eller.process(imgRGB)
    # print(results.multi_hand_landmarks)
    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            Cizim.draw_landmarks(img , handLms, El.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h,w, _ = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])


                # if id == 8:
                #     cv2.circle(img, (cx,cy), 9, (255,0,0), cv2.FILLED)
                # if id == 6:
                #     cv2.circle(img, (cx,cy), 9, (0,0,255), cv2.FILLED)

    if len(lmList) != 0:
        fingers = []
        for id in range(0,5): # 5 parmak oldugu icin id burada index gibi dusunulebilir
            if lmList[tipIds[id]][1] > lmList[tipIds[id]-2][1]: # o idnin yüksekliği önceki idnin yüksekliğinden büyükse
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        cv2.putText(img, str(totalFingers), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    print(lmList)

    cv2.imshow("img",img)
    cv2.waitKey(1)