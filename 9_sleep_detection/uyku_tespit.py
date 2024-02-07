import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture("video2.mp4")
detector = FaceMeshDetector()
plotY = LivePlot(540, 360, [10, 60])

counter = 0
blickCounter = 0
ratioList = []
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243] # sağdaki gözün noktaları
color = (0,0,255)

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img,draw = False) #draw = False ile noktaları göstermeyi kapatiyoruz

    if faces: # eğer yüz bulunursa
        face = faces[0]

        for id in idList:
            cv2.circle(img, face[id], 5, (0,0,255), cv2.FILLED)

        solUst = face[159]
        solAlt = face[23]
        solSol = face[130]
        solSag = face[243]

        lengthVer, _ = detector.findDistance(solUst, solAlt)
        lengthHor, _ = detector.findDistance(solSol, solSag)

        cv2.line(img, solUst, solAlt, (0,255,0),3)
        cv2.line(img, solSol, solSag, (0,255,0),3)

        ratio = int((lengthVer/lengthHor)*100)
        ratioList.append(ratio)
        if len(ratioList)>3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)
        print(ratioAvg)

        if ratioAvg < 35 and counter == 0:
            blickCounter += 1
            color = (0,255,0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (0,0,255)

        cvzone.putTextRect(img, f'Blink Count: {blickCounter}', (50,100), colorR = color)

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640,360))
        imgStack = cvzone.stackImages([img, imgPlot], 2,1)



    cv2.imshow("Image", imgStack)
    cv2.waitKey(25)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

