from tokenize import Imagnumber
import imagehash
from PIL import Image
import os

hashItems = []
pathOfItems = []#contains the list of non repeating images
path = './images'
for image in os.listdir(path):
    val = imagehash.average_hash(Image.open(str(path)+'/'+str(image)))
    if val not in hashItems:
        hashItems.append(val)
        pathOfItems.append(path+'/'+image)
for item in pathOfItems:
    print(item)