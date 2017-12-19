# Face Recognition
A Webapp to detect face of Narendra Modi and Arvind Kejriwal using convolutional Neural Network .

## Running Instructions

1. cd to the recognition directory
2. In console execute = python app.py
3. Upload an Image 
4. Check the results

## UI
![alt text](https://github.com/rishab-sharma/recognition/blob/master/Screen%20Shot%202017-12-19%20at%204.28.07%20PM.png)

## Objective
To develop a simple Web Application, which when given an image, is able to detect a face (show with a boundary box) and give a result saying whether Narendra Modi and/or Arvind Kejriwal are present in the image or not.

You are not allowed to use any online services and/or APIs for facial recognition, but are supposed to build your own model(s) from scratch. Feel free to use any image processing and learning techniques/libraries. You have the full liberty to decide on the architecture of the system.

You are required to curate a data set for your model(s) by using any image search APIs and would be required to store them in a MongoDB collection. We expect you to split your collected data set into training and testing sets. Use the training set only, to train your model(s) and then test the trained model on the testing data set. Make sure, there is no overlap of samples between the training and the testing data sets.

## Dependencies

bleach==1.5.0
click==6.7
enum34==1.1.6
Flask==0.12.2
gunicorn==19.7.1
h5py==2.7.1
html5lib==0.9999999
itsdangerous==0.24
Jinja2==2.10
Keras==2.1.2
Markdown==2.6.10
MarkupSafe==1.0
numpy==1.13.3
olefile==0.44
opencv-python==3.3.0.10
Pillow==4.3.0
protobuf==3.5.0.post1
pymongo==3.6.0
PyYAML==3.12
scipy==1.0.0
six==1.11.0
tensorflow==1.4.1
tensorflow-tensorboard==0.4.0rc3
Werkzeug==0.13

## Model

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
