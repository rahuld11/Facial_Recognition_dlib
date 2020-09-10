#Importing
#First we will import the relevant libraries
import cv2
import numpy as np
import dlib
import face_recognition
import os
from datetime import datetime

# Load the image path
path = 'G:/FaceRecognitionProject/ImagesAttendence'
images = []     # LIST CONTAINING ALL THE IMAGES
classNames = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS Names

myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    
print(classNames)
        
#Compute Encodings
def findEncodings (images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('G:/FaceRecognitionProject/Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encode Compute')

cap = cv2.VideoCapture(0)

while True:
    #Webcam Image
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    #Webcam Encodings
    facesCurFrame = face_recognition.face_locations(imgS) 
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    #Find Matches
    for encodeFace, faceLoc in zip (encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markAttendance(name)
            
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    
    #if cv2.waitKey(1) == 13:
        #break        
          
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
   
cap.release()
cv2.destroyAllWindows()

  
    