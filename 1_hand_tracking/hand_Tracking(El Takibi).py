# önce elimizin tespit edilmesi lazım ki daha sonra elimizin içindeki parmakların tespit edilsin

import cv2
import time
import mediapipe as mp # mediapipe google tarafından geliştirilmiş bir kütüphanedir

cap = cv2.VideoCapture(0) # buraya 0 yazmamızın sebebi bilgisayarda birden fazla kamera olabilir
                          # default olan her zaman 0 oluyor

mpHand = mp.solutions.hands # el tespiti yapiliyor

mpHands = mpHand.Hands(max_num_hands = 2) # maks kaç eli tespit edeceğimizi belirtiyoruz

mpDraw = mp.solutions.drawing_utils # eklem noktalarını çiziyor

pTime = 0
cTime = 0

#Önce detect edilir, daha sonra tracking sağlanır. detection, tracking'e göre daha yavaştır.

while True: # kameran calistigi surece
    success, kamera = cap.read() # kameradan okuma yapılıyor
    #önce resmi rgbye çevir
    imgRGB = cv2.cvtColor(kamera,cv2.COLOR_BGR2RGB)

    results = mpHands.process(imgRGB) # gelen kareyi el tespiti için işliyoruz
    print(results.multi_hand_landmarks) # elin içindeki noktaları yazdırıyoruz

    if results.multi_hand_landmarks: # eğer el tespit edilmişse
        for handLms in results.multi_hand_landmarks: # elin içindeki noktaları tek tek dolaşıyoruz
            mpDraw.draw_landmarks(kamera,handLms,mpHand.HAND_CONNECTIONS) # noktaları çiziyoruz

            for id, lm in enumerate(handLms.landmark): # noktaların id ve x,y,z değerlerini alıyoruz
                #print(id,lm)
                h,w,c = kamera.shape # kameranın yüksekliği, genişliği ve kanal sayısını alıyoruz

                cx,cy = int(lm.x*w) , int(lm.y*h) # noktaların x ve y değerlerini alıyoruz

                #bilek ( burada bileği tespit ediyoruz ve boş bir daire çiziyoruz)
                if id==0:
                    cv2.circle(kamera, (cx,cy),9,(255,0,0),cv2.FILLED)


    cTime = time.time() # şuanki zamanı alıyoruz
    fps = 1/(cTime-pTime) # fps hesaplıyoruz
    pTime = cTime # şuanki zamanı geçmiş zamana eşitliyoruz

    cv2.putText(kamera,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("kamera",kamera)
    cv2.waitKey(1)