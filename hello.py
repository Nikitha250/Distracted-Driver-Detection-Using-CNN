
from flask import Flask,render_template
from tkinter import *  #gui
import tkinter
import numpy as np
import imutils #video proce
#import dlib

import argparse
import sys
import cv2
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from keras.preprocessing import image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename



app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
	return render_template('home.html')




@app.route("/upload")
def upload():
    global video
    filename = filedialog.askopenfilename(initialdir="Video")
    #pathlabel.config(text="          Video loaded")
    video = cv.VideoCapture(filename)
    return render_template('home.html')



@app.route("/startMonitoring")   
def startMonitoring():
			while(True):
				ret, frame = video.read()
				print(ret)
				if ret == True:
					cv.imwrite("test.jpg",frame)
					imagetest = image.load_img("test.jpg", target_size = (150,150))
					imagetest = image.img_to_array(imagetest)
					imagetest = np.expand_dims(imagetest, axis = 0)
					predict = awgrd_model.predict_classes(imagetest)
					print(predict)
					msg = "";
					if str(predict[0]) == '0':
							msg = 'Safe Driving'
					if str(predict[0]) == '1':
							#msg = 'Using/Talking Phone'
							msg = 'Texting right'
					if str(predict[0]) == '2':
							msg = 'Talking On phone Right'
					if str(predict[0]) == '3':
							#msg = 'Using/Talking Phone'
							msg = 'texting Left'
					if str(predict[0]) == '4':
							#msg = 'Using/Talking Phone'
							msg = 'Talking Phone left'
					if str(predict[0]) == '5':
							#msg = 'Drinking/Radio Operating'
							msg='Operating the Radio/Texting'
					if str(predict[0]) == '6':
							#msg = 'Drinking/Radio Operating'
							msg = 'Drinking'
					if str(predict[0]) == '7':
							msg = 'Reaching Behind'
					if str(predict[0]) == '8':
							msg = 'Hair & Makeup'
					if str(predict[0]) == '9':
							msg = 'Talking To Passenger'
					text_label = "{}: {:4f}".format(msg, 85)
					cv.putText(frame, text_label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
					cv.imshow('Frame', frame)
					if cv.waitKey(2500) & 0xFF == ord('q'):
						break
					else:
						break
				video.release()
				cv.destroyAllWindows()
				
			return render_template('home.html')



    


@app.route("/loadModel")
def loadModel():
			global awgrd_model
			img_width, img_height = 150, 150
			train_data_dir = 'dataset/imgs/train'
			validation_data_dir = 'dataset/imgs/validation'
			nb_train_samples = 22424
			nb_validation_samples = 1254
			nb_epoch = 10

			if os.path.exists('AWGRD_model.h5'):
				awgrd_model = Sequential()
				awgrd_model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
				awgrd_model.add(MaxPooling2D(pool_size = (2, 2)))
				awgrd_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
				awgrd_model.add(MaxPooling2D(pool_size = (2, 2)))
				awgrd_model.add(Flatten())
				awgrd_model.add(Dense(output_dim = 128, activation = 'relu'))
				awgrd_model.add(Dense(output_dim = 10, activation = 'softmax'))
				awgrd_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
				awgrd_model.load_weights('AWGRD_model.h5')
				print(awgrd_model.summary())
				#pathlabel.config(text="          AWGRD Model Generated Successfully")
			else:
				awgrd_model = Sequential()
				awgrd_model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
				awgrd_model.add(MaxPooling2D(pool_size = (2, 2)))
				awgrd_model.add(Convolution2D(32, 3, 3, activation = 'relu'))
				awgrd_model.add(MaxPooling2D(pool_size = (2, 2)))
				awgrd_model.add(Flatten())
				awgrd_model.add(Dense(output_dim = 128, activation = 'relu'))
				awgrd_model.add(Dense(output_dim = 10, activation = 'softmax'))
				awgrd_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
				train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
				test_datagen = ImageDataGenerator(rescale=1.0/255)
				train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical')
				validation_generator = train_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical')
				awgrd_model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, validation_data=validation_generator, nb_val_samples=nb_validation_samples)
				awgrd_model.save_weights('driver_state_detection_small_CNN.h5')
				#pathlabel.config(text="          AWGRD Model Generated Successfully")
			return render_template('home.html')




if __name__ == '__main__':

	app.run(debug=True)
