import cv2

def getDataFromFrame(frame, mpHands, mpFace):
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    hands = mpHands.process(frameRGB)
    face = mpFace.process(frameRGB)

    if hands.multi_hand_landmarks and face.multi_face_landmarks:
        return hands.multi_hand_landmarks, face.multi_face_landmarks
    elif hands.multi_hand_landmarks:
        return hands.multi_hand_landmarks, None
    elif face.multi_face_landmarks:
        return None, face.multi_face_landmarks
    else:
        return None, None
    
def processFrame(frame, mpHands, mpFace):
    coords = []

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    results = mpHands.process(frameRGB)
    face = mpFace.process(frameRGB)
    
    handCoords = []
    centerFaceArea = [0, 0]

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) > 2:
            return None

        for handLandmarks in results.multi_hand_landmarks:
            for landmark in handLandmarks.landmark:
                handCoords.append([int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])])
            
    if face.multi_face_landmarks:
        centerFaceArea = [int(face.multi_face_landmarks[0].landmark[10].x * frame.shape[1]), int(face.multi_face_landmarks[0].landmark[10].y * frame.shape[0])]
    else:
        centerFaceArea = [frame.shape[1]//2, frame.shape[0]//5]    

    for coord in handCoords:
        coords.append(coord[0] - centerFaceArea[0])
        coords.append(coord[1] - centerFaceArea[1])

    return coords