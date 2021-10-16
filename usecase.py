import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import cv2

# configs variables
width = 600
height = 600
IMGsize = (width, height)
texts = ['horizontal "_"', 'vertical "|"']
font = cv2.FONT_HERSHEY_SIMPLEX # font
org = (00, 185) # org
fontScale = 1   # fontScale
color = (0, 0, 255) # Red color in BGR
thickness = 2   # Line thickness of 2 px



def create_model():
    model = tf.keras.models.Sequential()
    #CNN
    model.add(tf.keras.layers.Conv2D(32,3, activation='relu', input_shape=(width,height,3)))
    model.add(tf.keras.layers.Conv2D(32,3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))

    model.add(tf.keras.layers.Conv2D(16,3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(16,3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))

    model.add(tf.keras.layers.Conv2D(8,3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(8,3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))

    model.add(tf.keras.layers.Flatten())
    # ANN
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    return model


model = create_model()



def prediction(path):
    img = image.load_img(path, target_size=IMGsize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.
    classes = model.predict(x, batch_size=10)
    return classes  #[[h,v]]


def predict_cv(img):
    x = cv2.resize(img, IMGsize, interpolation=cv2.INTER_AREA)
    x = np.expand_dims(x, axis=0) / 255.
    classes = model.predict(x)
    return classes  #[[h,v]]




def main():
    vc = cv2.VideoCapture(0)
    checkpoint_path = "./training_1/cp.ckpt"
    # Loads the weights
    model.load_weights(checkpoint_path)
    # capture loop
    while True:        
        _, img = vc.read()
        
        classes = predict_cv(img)
        
        if classes[0][0] > 0.5:
            img = cv2.putText(img, texts[0], org, font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)
            print("horizontal")
        else:
            img = cv2.putText(img, texts[1], org, font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)
            print("vertical")
            
        cv2.imshow("Capture", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # click on q button to quit the windows
            break



    vc.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()