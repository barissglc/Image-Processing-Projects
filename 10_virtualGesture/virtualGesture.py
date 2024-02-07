# Kütüphanelerimizi içe aktaralım
import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8)

startDistance = None
lmList1 = None
lmList2 = None
scale = 0
cx, cy = 0, 0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    img1 = cv2.imread("mikroplar.png")

    # Görüntüyü yeniden boyutlandırma
    img1 = cv2.resize(img1, (128, 128))



    if len(hands) == 2:
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]
            if startDistance is None:
                # length, info, img = detector.findDistance(lmList1[8][:2], lmList2[8][:2], img)
                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
                startDistance = length

            length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)
            scale = int((length - startDistance) // 2)
            cx,cy = info[4:]
            print(scale)
    else:
        startDistance = None

    try:
        h1, w1, _ = img1.shape
        newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
        img1 = cv2.resize(img1, (newW, newH))

        img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
    except:
        pass

    cv2.imshow("Ekran", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
