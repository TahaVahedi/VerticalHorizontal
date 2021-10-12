# preprocessing and create vertical set images (rotate 90 degrees)
import os
from glob import glob as g 
import cv2


# train test split
#download your images and put them in same folder and update the original_dataset 
original_dataset = "./landspace_dataset_original/"  # here i use https://www.kaggle.com/arnaud58/landscape-pictures
all_imges = g(original_dataset + "*.jpg")
n_images = int(len(all_imges))
print(n_images)
train_images = all_imges[:int(n_images - n_images*0.1)]
test_images = all_imges[int(n_images - n_images*0.1):]
print(len(train_images),len(test_images)) 
print(len(train_images)+len(test_images)) 
destination_path = "./dataset/train/"
for i in range(len(train_images)):
    os.rename(train_images[i], destination_path+str(i)+".jpg")
destination_path = "./dataset/test/"
for i in range(len(test_images)):
    os.rename(test_images[i], destination_path+str(i)+".jpg")
# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")   # move a file



# preprocessing images

width = 600
height = 600
dim = (width, height)

def saveIMG(image, label, name, train="True"):
    if train == "True":
        file_name = "./dataset/train/{}/{}".format(label, name)
    else:
        file_name = "./dataset/test/{}/{}".format(label, name)

    cv2.imwrite(file_name, image)




# train set ======

train_path = "./dataset/train/"
all_train_images = g(str(train_path)+"*.jpg")

print("start processing...")

for src in all_train_images:
    name1 = str(src.split("\\")[-1])
    name = train_path+name1

    img1 = cv2.resize(cv2.imread(name ,cv2.IMREAD_COLOR), dim, interpolation=cv2.INTER_AREA)
    saveIMG(img1, "horizontal", name1, train="True")
    
    img2 = cv2.rotate(img1, cv2.cv2.ROTATE_90_CLOCKWISE) 
    saveIMG(img2, "vertical", name1, train="True")
    os.remove(name)
    

print("train set Done!")   


# test set ==========

test_path = "./dataset/test/"
all_test_images = g(str(test_path)+"*.jpg")



for src in all_test_images:
    name1 = str(src.split("\\")[-1])
    name = train_path+name1

    img1 = cv2.resize(cv2.imread(name ,cv2.IMREAD_COLOR), dim, interpolation=cv2.INTER_AREA)
    saveIMG(img1, "horizontal", name1, train="null")
    
    img2 = cv2.rotate(img1, cv2.cv2.ROTATE_90_CLOCKWISE) 
    saveIMG(img2, "vertical", name1, train="null")
    os.remove(name)
    

print("test set Done!")   
