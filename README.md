# Face Recognition
A Webapp to detect face of Narendra Modi and Arvind Kejriwal using convolutional Neural Network .

## Running Instructions

1. cd to the recognition directory
2. in console execute = python app.py
3. Upload an Image 
4. Check the results

## UI
![alt text](https://github.com/rishab-sharma/recognition/blob/master/Screen%20Shot%202017-12-19%20at%204.28.07%20PM.png)

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
