import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0) # Get the input video from default camera
detector = HandDetector(maxHands=1, detectionCon=0.8)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") # Use classifier class and get model files and labels from Models directory to detect hand gestures

offset = 20 # Set the offset value to provide padding when cropping the image
whiteBgSize = 300 # Set background size for imgWhite
imgWhite = np.ones((whiteBgSize, whiteBgSize, 3), np.uint8) # Defined the value of imgWhite us numpy module

labels = ["A", "B", "C"] # Variable which save the labels for print on detector

while True:
    success, img = cap.read()
    imgOutput = img.copy() # Copied the frame or value of img
    hands, img = detector.findHands(img) # Detected there a hand on the input

    if hands: # If a hand is detected
        hand = hands[0] # Get the data from detected hand
        x, y, w, h = hand['bbox'] # Break down the bounding box value into variables x, y, w, h

        # Change the color of imgWhite to white
        imgWhite = np.ones((whiteBgSize, whiteBgSize, 3), np.uint8) * 255  # Used for cropped img background
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the image according to the value on the bbox

        imgCropShape = imgCrop.shape  # Get the array dimension of imgCrop (width, height, color)

        aspectRatio = h / w  # Get a some value for checking imgCrop ratio

        # Checking whether the length of the bbox height is longer than the width
        if aspectRatio > 1:  # If the length of the bounding box height is longer then the width
            const = whiteBgSize / h  # Get a value for change a width of imgCrop
            wCalculate = math.ceil(const * w)  # Calculates the new width of imgCrop
            imgResize = cv2.resize(imgCrop, (wCalculate, whiteBgSize))  # Resize the imgCrop (width : wCalculate, height : whiteBgSize)
            imgResizeShape = imgResize.shape  # Get the array dimension of imgResize (width, height, color)
            wGap = math.ceil((whiteBgSize - wCalculate) / 2)  # Calculates the difference width of whiteBgSize and wCalculate(new width)
            imgWhite[:,wGap:wCalculate + wGap] = imgResize  # Positioning the imgResize on the center of imgWhite(background)
            prediction, index = classifier.getPrediction(imgWhite, draw=False) # Hand shape prediction whose value is stored in the imgWhite variable according to the existing model
        else: # If the length of the bounding box width is longer then the height
            const = whiteBgSize / w  # Get a value for change a height of imgCrop
            hCalculate = math.ceil(const * h)  # Calculates the new height of imgCrop
            imgResize = cv2.resize(imgCrop, (whiteBgSize, hCalculate))  # Resize the imgCrop (width : whiteBgSize, height : hCalculate)
            imgResizeShape = imgResize.shape  # Get the array dimension of imgResize (width, height, color)
            hGap = math.ceil((whiteBgSize - hCalculate) / 2)  # Calculates the difference height of whiteBgSize and hCalculate(new height)
            imgWhite[hGap:hCalculate + hGap,:] = imgResize  # Positioning the imgResize on the center of imgWhite(background)
            prediction, index = classifier.getPrediction(imgWhite, draw=False) # Hand shape prediction whose value is stored in the imgWhite variable according to the existing model

        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED) # Create and set a bbox for text labels
        cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2) # set text position of labels on center of labels bounding box
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4) # Create and set a bounding box on detected hand

        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
