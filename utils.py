import time
import os
import cv2

import h5py
import tensorflow as tf
import numpy as np
way = os.path.dirname(os.path.abspath(__file__))
def face_extraction(f_cascade, input_img, scaleFactor = 1.1):
    img = input_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    for (a, b, c, d) in arr:
        cv2.rectangle(img, (a, b), (a+c, b+d), (0, 255, 0), 2)
    return len(arr), img

from keras.models import load_model
def cnn_models():
	cnn_modi = load_model(os.path.join(way,"models/cnn_modi.h5"))
	cnn_kejriwal = load_model(os.path.join(way,"models/cnn_kejriwal.h5"))
	cnn_modi.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	cnn_kejriwal.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	graph = tf.get_default_graph()
	return cnn_kejriwal, cnn_modi, graph

from keras.preprocessing import image	

def predictor(path, model):
	img_inp = image.load_img(path, target_size=(150,150))
	p = image.img_to_array(img_inp)
	p = np.expand_dims(p, axis=0)
	array = np.vstack([p])
	classes = model.predict_classes(array, batch_size=10, verbose=0)
	if (classes[0][0]) == 1:
		return True
	else:
		return False
