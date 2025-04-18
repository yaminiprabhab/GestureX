import cv2
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 50

# Load toolbar images
folderPath = "C:/Users/Yamini Prabha/OneDrive/Desktop/OpenCV/GestureX/Header"
myList = os.listdir(folderPath)
overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]

header = overlayList[0]  # Default header
drawColor = (128, 0, 128)  # Set default to Purple

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index Finger Tip
        x2, y2 = lmList[12][1:] # Middle Finger Tip

        fingers = detector.fingersUp()

        # Selection Mode (Index + Middle Finger Up)
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:  
            if y1 < 120:
                if 668 < x1 < 780:
                    header = overlayList[0]
                    drawColor = (128, 0, 128)  
                elif 815 < x1 < 923:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)  
                elif 956 < x1 < 1065:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)  
                elif 1115 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)  # Eraser

            xp, yp = 0, 0  # Reset previous points to fix unwanted lines
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Erasing Mode (Full Hand Open)
        elif all(fingers):  
            cv2.circle(img, (x1, y1), 50, (0, 0, 0), cv2.FILLED)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
            xp, yp = x1, y1

        # Drawing Mode (Only Index Finger Up)
        elif fingers[1] and not fingers[2]:  
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1  # Start new stroke
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0  # Reset points when fingers are lifted

    # Merge layers
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:120, 0:1280] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
