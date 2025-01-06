import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import cv2
import mediapipe as mp
from Modules import utils
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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

def processVideo(videoURL):
    probabilities = [0 for x in range(numClasses)]
    sentence = []

    cap = cv2.VideoCapture(videoURL)
    if not cap.isOpened(): return -1
    
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break

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
                sentence.append(i)
                probabilities = [0 for x in range(numClasses)]
                probabilities[i] =- 99999
                break

    cap.release()
    cv2.destroyAllWindows()

    processedSentence = [] 
    for x in sentence:
        word = dictClasses[x]
        word = word.replace("_INV_", "")
        processedSentence.append(word)
    
    return(" ".join(processedSentence))
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processVideo', methods=['GET'])
def processVideoAPI():
    VideoURL = request.args.get('VideoURL')
    sentence = processVideo(VideoURL)

    if sentence == -1:
        return jsonify({
            "text": "Video not found",
            "status_code": 404
        })

    if len(sentence) == 0:
        return jsonify({
            "text": "No sentence detected",
            "status_code": 404
        })
    
    return jsonify({
        "text": sentence,
        "status_code": 200
    })

if __name__ == '__main__':
    app.run(debug=True)
