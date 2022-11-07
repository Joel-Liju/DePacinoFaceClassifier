import cv2
import fnmatch
import imagehash
from PIL import Image
# from matplotlib import pyplot as plt
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#make sure to have a finalImages folder and a images folder.
hashItems = []
pathOfItems = []#contains the list of non repeating images
finalPaths = ['./images/Robert_De/','./images/Al_Pacino/']
paths = ['./imgs/Robert_De','./imgs/Al_Pacino']
for index,path in enumerate(paths):
    for root,_,files in os.walk(path):
        for filename in files:
            file = os.path.join(root,filename)
            if fnmatch.fnmatch(file,"*.png"):
                img = cv2.imread(file)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray,1.1,4)
                for (x,y,w,h) in faces:
                    crop_face = img[y:y+h,x:x+w]
                # print(crop_face)
                # path = os.path.join(root,filename)
                if len(faces)==1:
                    cv2.imwrite(finalPaths[index]+filename,crop_face)
                # print(finalPaths[index]+filename)

# for image in os.listdir(path):
#     val = imagehash.average_hash(Image.open(str(path)+'/'+str(image)))
#     img = cv2.imread(path+'/'+image)

#     grayscaledImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(grayscaledImg, 1.1, 3)
#     #if val not in hashItems and len(faces)==1:#the same images is not already there and the number of faces is 1
#     hashItems.append(val)
#     pathOfItems.append(path+'/'+image)
#     cv2.imwrite("./finalImages/"+image,img)