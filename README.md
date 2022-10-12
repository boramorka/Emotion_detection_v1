<h1 align="center">
  <br>
 Face Expression Recognition + Live Webcam Detection <br>
 :smiley: :neutral_face: :unamused: :movie_camera:
</h1>


<h3 align="center">
  Built with
  <br>
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" height="30">
    <img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/tensorflow.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/matplotlib.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/numpy.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/pandas.svg" height="30">
    <img src="https://raw.githubusercontent.com/boramorka/usercontent/aad4d15178483720bcc0562617c86a7c84a7d257/shields.io/scikit-learn.svg" height="30">
    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=green" height="30">
</h3>

<p align="center">
  <a href="#how-to-run-locally">How To Run Locally</a> •
  <a href="#built-process">Built process</a> •
  <a href="#feedback">Feedback</a>
</p>

## How To Run Locally

  ``` bash
  # Clone this repository
  $ git clone https://github.com/boramorka/Emotion_detection_v1.git

  # Go into the repository
  $ cd Emotion_detection_v1

  # Install dependencies
  $ pip install requirements.txt

  # Run app
  $ python main.py
  ```

### Make sure you have:
- Nvidia videocard
- Installed all requirement libraries
- Installed Nvidia Cuda
- Installed Nvidia drivers
- Installed cuDNN
  
![Usage](https://github.com/boramorka/usercontent/blob/main/face-detection/face-detection.gif?raw=true)

## Built process

- Dateset for training: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

- Import libraries:
  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import seaborn as sns
  import os

  # Importing Deep Learning Libraries

  from keras.preprocessing.image import load_img, img_to_array
  from keras.preprocessing.image import ImageDataGenerator
  from keras.layers import Dense,Input,Dropout,GlobalAveragePooling2D,Flatten,Conv2D,BatchNormalization,Activation,MaxPooling2D
  from keras.models import Model,Sequential
  from keras.optimizers import Adam,SGD,RMSprop
  ```
- Model architecture:
  

  ```python
  model = Sequential()

  #1st CNN layer
  model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (48,48,1)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(0.25))

  #2nd CNN layer
  model.add(Conv2D(128,(5,5),padding = 'same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout (0.25))

  #3rd CNN layer
  model.add(Conv2D(512,(3,3),padding = 'same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout (0.25))

  #4th CNN layer
  model.add(Conv2D(512,(3,3), padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())

  #Fully connected 1st layer
  model.add(Dense(256))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  # Fully connected layer 2nd layer
  model.add(Dense(512))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))

  model.add(Dense(no_of_classes, activation='softmax'))

- The model has a 4.5 million parameters

- Now we can recognize 7 emotions 
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

- Creating main.py for capturing:
  ```python
  """Capturing block"""

  cap = cv2.VideoCapture(0)

  while True:
      _, frame = cap.read()
      labels = []
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      faces = face_classifier.detectMultiScale(gray)

      main_model()

      cv2.imshow('Emotion Detector',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  ```

- Main model for face detection and putting text:
  ```python
    def main_model():
      for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    ```




## Feedback
:person_in_tuxedo: Feel free to send me feedback on [Telegram](https://t.me/boramorka). Feature requests are always welcome. 

:abacus: [Check my other projects.](https://github.com/boramorka)


