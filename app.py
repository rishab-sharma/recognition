import os
import time
from flask import Flask, render_template, request
from werkzeug import secure_filename
import tensorflow as tf
import cv2
import numpy as np
from utils import face_extraction, cnn_models, predictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model_ak, model_nm, graph = cnn_models()
haar_face_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT,'models/data/haarcascade_frontalface_alt.xml'))

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/", methods=['POST'])
def process():
	target = os.path.join(APP_ROOT, 'static/images/')
	if not os.path.isdir(target):
		os.mkdir(target)
	file = request.files['file']
	filename = file.filename
	destination = os.path.join(APP_ROOT, 'static/images/')+str(secure_filename(file.filename))
	file.save(destination)
	img = cv2.imread(os.path.join(APP_ROOT, destination))
	n_faces, faces_detected_img = face_extraction(haar_face_cascade, img)
	faces="No"
	AK="No"
	NM="No"
	if n_faces > 0:
		faces = "Yes"
		with graph.as_default():
			if (predictor(destination, model_ak) == True):
				AK = "Yes"
			if (predictor(destination, model_nm) == True):
				NM = "Yes"
	cv2.imwrite(destination, faces_detected_img)
	source = '/static/images/'+file.filename
	return render_template("index.html",faces=faces, AK=AK, NM=NM, source=source)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
