import cv2
from tokenize import Imagnumber
import imagehash
from PIL import Image
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#make sure to have a finalImages folder and a images folder.
hashItems = []
pathOfItems = []#contains the list of non repeating images
path = './images'
for image in os.listdir(path):
    val = imagehash.average_hash(Image.open(str(path)+'/'+str(image)))
    img = cv2.imread(path+'/'+image)

    grayscaledImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayscaledImg, 1.1, 3)
    #if val not in hashItems and len(faces)==1:#the same images is not already there and the number of faces is 1
    hashItems.append(val)
    pathOfItems.append(path+'/'+image)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
    cv2.imwrite("./finalImages/"+image,img)