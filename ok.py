from base64 import encode
from operator import le
import re
import cv2
from cv2 import COLOR_BGR2RGB
import face_recognition
import os
import numpy as np
from datetime import datetime 
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


path = 'img'
img = []
bo_anh = []
list_img = os.listdir(path)

for list in list_img:
    curImg = cv2.imread(f"{path}/{list}")
    re_anh = cv2.resize(curImg,(500,500))
    img.append(curImg)
    bo_anh.append(os.path.splitext(list)[0])



def Mahoa(img):
    ecd_img = []
    for ecd in img:
        ecd = cv2.cvtColor(ecd, cv2.COLOR_RGB2BGR)
        #encode = face_recognition.face_locations(ecd)[0]
        encode = face_recognition.face_encodings(ecd)[0] 
        ecd_img.append(encode)  
    return ecd_img   


def Attendance(name,valmin):
    with open('text.txt','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        #if name not in nameList:
        now = datetime.now()
        dtString = now.strftime('%d/%m/%Y, %H:%M:%S')
        f.writelines(f'\n{name},{dtString},{valmin}') 
    
encodeLisKnow = Mahoa(img)  

   

cap = cv2.VideoCapture('mevideo.mp4')

while True: 
    ret, frame = cap.read() #ret: load đc trả về True, sai Faile
    frames = cv2.resize(frame,(0,0),None,fx=1,fy=1)
    frames = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    location_fr = face_recognition.face_locations(frames)
    ecding_fr = face_recognition.face_encodings(frames)

    for ecFace, locaFace in zip(ecding_fr, location_fr): #lay tugng khuon mat va vi tri theo cặp
        match = face_recognition.compare_faces(encodeLisKnow,ecFace) #đối chieu
        faceDis = face_recognition.face_distance(encodeLisKnow,ecFace) #distance: khoảng cách
        matchIndex = np.argmax(faceDis)


        if match[matchIndex]>0.5:
            name = bo_anh[matchIndex].upper()
            valmin = "{}".format(round(100*(1-faceDis[matchIndex])))
        else:
            name = "Unknow"

        #print tên lên frame
        y1, x2, y2, x1 = locaFace
        #y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
        cv2.rectangle(frame,(x1,y1), (x2,y2),(0,0,255),2)
        cv2.putText(img,name + ' - ' + valmin +'%',(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
        Attendance(name,valmin)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key ==27:
        break
	


       
cap.release()
cv2.destroyAllWindows()
    
