#IMPORTING THE PACKAGES 
import cv2                       #for capturing video and adding graphics
import face_recognition          #for recognising different faces
import pyttsx3                   #text to speech module
import numpy as np               #geting minimum value
import os                        #for file function   
import datetime                  #getting current time

#ASSIGNING VARIABLES
engine = pyttsx3.init()
imgPath = "C:\\Users\\manis\\Desktop\\Python Programs\\Grade 10\\CV2\\Image_Attendence"
images = []
class_student_names = []
myList = os.listdir(imgPath)
print("TOTAL PHOTOS:",len(myList),":", myList)

#getting names for class students
for names in myList:
    curImg = cv2.imread(f'{imgPath}/{names}')
    images.append(curImg)
    class_student_names.append(os.path.splitext(names)[0])
print("CLASS STUDENT NAMES:",class_student_names)

#function for encoding all the images
def findEncodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

#function for marking attendence
def markAttendence(name):
    with open('C:\\Users\\manis\\Desktop\\Python Programs\\Grade 10\\AI_BOOTCAMP\\AttendenceData.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        now = datetime.datetime.now()
        entry_time = now.replace(hour=21, minute=42, second=0, microsecond=0)

        if name not in nameList:
            engine.say("GOOD MORNING", name)
            engine.runAndWait()
            if now < entry_time:
                o_l = "On Time"
                now = datetime.datetime.now()
                dt = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dt}, {o_l}')
            else:
                if (now > entry_time):
                    o_l = "Late"
                    now = datetime.datetime.now()
                    dt = now.strftime('%H:%M:%S')
                    f.writelines(f'\n{name},{dt},{o_l}')
                

encodeListKnown = findEncodings(images)
print("ENCODING COMPLETED...")

#starting the camera
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    imgCAP = cv2.resize(img , (0,0), None, 0.25,0.25)
    imgCAP = cv2.cvtColor(imgCAP, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgCAP)
    encodeCurFrame = face_recognition.face_encodings(imgCAP, facesCurFrame)

    for encodeFace,faceloc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matches_Index = np.argmin(faceDis)
        
        if matches[matches_Index]:
            name = class_student_names[matches_Index].upper()
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-30),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img, name,(x1+15, y2-6), cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),1)
            markAttendence(name)
    
    cv2.imshow("WEBCAM", img)
    if cv2.waitKey(1) & 0xFF == 27: #Stoping the code after pressing escape key
        break
cap.release()
cv2.destroyAllWindows()
