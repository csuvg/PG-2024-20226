import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import cv2
import mediapipe as mp
from Modules import utils
import os

with open('Data/modelParams.pkl', 'rb') as f:
    numClasses = pickle.load(f)
    dictClasses = pickle.load(f)
    minPoint = pickle.load(f)
    maxPoint = pickle.load(f)
    confidence = pickle.load(f)
    threshold = pickle.load(f)

model = tf.keras.models.load_model('Models/modelDropoutBN.keras')

mpHands = mp.solutions.hands.Hands()
mpFace = mp.solutions.face_mesh.FaceMesh()
mpDrawing = mp.solutions.drawing_utils

def processVideo(video):
    probabilities = [0 for x in range(numClasses)]
    sentence = []

    cap = cv2.VideoCapture(video)
    
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break

        if len(sentence) > 0:
            sentenceStr = ' '.join(sentence)
            textSize = cv2.getTextSize(sentenceStr, cv2.FONT_HERSHEY_SIMPLEX, 4, 5)[0]
            textX = (frame.shape[1] - textSize[0]) // 2
            cv2.putText(frame, sentenceStr, (textX, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv2.LINE_AA)

        imageRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imageRgb.flags.writeable = False
        results = mpHands.process(imageRgb)
        imageRgb.flags.writeable = True
        imageBgr = cv2.cvtColor(imageRgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(imageBgr, handLandmarks, mp.solutions.hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', imageBgr)

        coords = utils.processFrame(frame, mpHands, mpFace)

        if coords is None or len(coords) < model.input_shape[1]:
            continue

        newCoords = []
        for coord in coords:
            if coord < 0: 
                newCoords.append(coord / abs(minPoint))
            elif coord > 0: 
                newCoords.append(coord / abs(maxPoint))
            else: 
                newCoords.append(0)

        coords = np.array(newCoords)
        coords = coords.reshape(1, coords.shape[0])

        prediction = model.predict(coords, verbose=0)
        if np.max(prediction) > confidence:
            predictedLabel = np.argmax(prediction)
            probabilities[predictedLabel] += 1

        for i in range(len(probabilities)):
            if probabilities[i] >= threshold:
                sentence.append(dictClasses[i].replace("_INV_", ""))
                probabilities = [0 for x in range(numClasses)]
                probabilities[i] =- 99999
                break

                # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            return -1

    cap.release()
    cv2.destroyAllWindows()

for video in os.listdir('Videos/Test/Normal'):
    res = processVideo('Videos/Test/Normal/' + video)

    if res == -1:
        break   
