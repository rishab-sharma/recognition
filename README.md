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
