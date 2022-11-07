from ctypes import util
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
import fnmatch
import os
import math
from ML.training import Flatten,normalize
# from matplotlib import pyplot as plt
from PIL import Image
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
resnet = InceptionResnetV1(pretrained='vggface2').eval()
# Load the cascade
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

def face_match(img_path, data_path): # img_path= location of photo, data_path= location of model you are created.pt 
    # getting embedding matrix of the given img
    global face_cascade,resnet
    labels = ["Al Pacino","Robert De Niro"]
    data_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Resize((224,224)),
         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
         transforms.RandomRotation(5, resample=False,expand=False, center=None),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    saved_data = torch.load(data_path)
    # Draw rectangle around the faces
    # crop_face = None
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for (x, y, w, h) in faces:
        crop_face = img[y:y+h, x:x+w]
        crop_face = data_transform(crop_face)
        crop_face = crop_face.unsqueeze(0)
        output = saved_data(crop_face)
        value, idx = torch.max(output,1)
        value2,idx2 = torch.min(output,1)
        print(value)
        if abs(value2-value)<0.06:
            print("neither")
        else:
            print(labels[idx.item()])
        cv2.imwrite("./test.png",img[y:y+h, x:x+w])
    # emb = resnet(img.unsqueeze(0)).detach() # detech is to make required gradient false
    # print(emb)
    # img = torch.from_numpy(img)
    
    # print(img.unsqueeze(0))
    # emb = resnet(img.unsqueeze(0)).detach() # detech is to make required gradient false
    
    # saved_data = torch.load(data_path) # loading data.pt file
    # print(saved_data)
    
    # saved_data.eval()
    # # print(saved_data[0])
    # # print(data)
    # # input, labels = DataLoader(img)
    # # input = 
    # # print(input)
    # output = saved_data(img)
    # print(output)
    # for idx, emb_db in enumerate(embedding_list):
    #     dist = torch.dist(emb, emb_db).item()
    #     dist_list.append(dist)
        
    # idx_min = dist_list.index(min(dist_list))
    # return (name_list[idx_min], min(dist_list))


result = face_match('', './model.pt')
# print('Face matched with: ',result[0], 'With distance: ',result[1])