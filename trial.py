from tkinter import *  #gui
from flask import Flask,render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
import os
from wtforms.validators import InputRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
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



#main = tkinter.Tk()
#main.title("Video-Based Abnormal Driving Behavior Detection")
#main.geometry("800x500")-->

global awgrd_model
global video

@app.route("/")
@app.route("/home")
def home():
      return render_template('home.html')  

@app.route("/livecam")
def livecam():

        
        # construct the argument parser and parse the arguments
        # ap = argparse.ArgumentParser()
        # ap.add_argument("-m", "--model", required=True,
        #         help="path to trained human activity recognition model")
        # ap.add_argument("-c", "--classes", required=True,
        #         help="path to class labels file")
        # ap.add_argument("-i", "--input", type=str, default="",
        #         help="optional path to video file")
        # args = vars(ap.parse_args())

        # load the contents of the class labels file, then define the sample
        # duration (i.e., # of frames for classification) and sample size
        # (i.e., the spatial dimensions of the frame)
        CLASSES = open("action_recognition_kinetics.txt").read().strip().split("\n")
        SAMPLE_DURATION = 16
        SAMPLE_SIZE = 112

        # load the human activity recognition model
        print("[INFO] loading human activity recognition model...")
        net = cv2.dnn.readNet("resnet-34_kinetics.onnx") 

        # grab a pointer to the input video stream
        print("[INFO] accessing video stream...")
        vs = cv2.VideoCapture(0) 

        # loop until we explicitly break from it
        while True:
                # initialize the batch of frames that will be passed through the
                # model
                frames = []

                # loop over the number of required sample frames
                for i in range(0, SAMPLE_DURATION):
                        # read a frame from the video stream
                        (grabbed, frame) = vs.read()

                        # if the frame was not grabbed then we've reached the end of
                        # the video stream so exit the script
                        if not grabbed:
                                print("[INFO] no frame read from stream - exiting")
                                sys.exit(0)

                        # otherwise, the frame was read so resize it and add it to
                        # our frames list
                        frame = imutils.resize(frame, width=1200)
                        frames.append(frame)

                # now that our frames array is filled we can construct our blob
                blob = cv2.dnn.blobFromImages(frames, 1.0,
                        (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                        swapRB=True, crop=True)
                blob = np.transpose(blob, (1, 0, 2, 3))
                blob = np.expand_dims(blob, axis=0)

                # pass the blob through the network to obtain our human activity
                # recognition predictions
                net.setInput(blob)
                outputs = net.forward()
                label = CLASSES[np.argmax(outputs)]

                # loop over our frames
                for frame in frames:
                        # draw the predicted activity on the frame
                        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
                        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)

                        # display the frame to our screen
                        cv2.imshow("Activity Recognition", frame)
                        key = cv2.waitKey(1) & 0xFF

                        # if the `q` key was pressed, break from the loop
                        if cv.waitKey(2500) & 0xFF == ord('q'):
                                break


@app.route('/loadModel')
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

    return render_template("home.html")
    
class UploadFileForm(FlaskForm):
    file1 = FileField("File", validators=[InputRequired()])
 
@app.route('/upload')   
def upload():
    global video
    form = UploadFileForm()
    #if form.validate_on_submit():
    file1 = form.file1.data
    #filename = filedialog.askopenfilename(initialdir="Video")
    #pathlabel.config(text="          Video loaded")
    video = cv.VideoCapture(file1)

    return render_template("home.html")



@app.route('/startMonitoring')
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
                        msg = 'Using/Talking Phone'
                if str(predict[0]) == '2':
                        msg = 'Talking On phone'
                if str(predict[0]) == '3':
                        msg = 'Using/Talking Phone'
                if str(predict[0]) == '4':
                        msg = 'Using/Talking Phone'
                if str(predict[0]) == '5':
                        msg = 'Drinking/Radio Operating'
                if str(predict[0]) == '6':
                        msg = 'Drinking/Radio Operating'
                if str(predict[0]) == '7':
                        msg = 'Reaching Behind'
                if str(predict[0]) == '8':
                        msg = 'Hair & Makeup'
                if str(predict[0]) == '9':
                        msg = 'Talking To Passenger'
                text_label = "{}: {:4f}".format(msg, 80)
                cv.putText(frame, text_label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.imshow('Frame', frame)
                if cv.waitKey(2500) & 0xFF == ord('q'):
                   break
            else:
                break
        video.release()
        cv.destroyAllWindows()

        return " started"


def exit():
    global main
    main.destroy()
  
'''
font = ('times', 16, 'bold')
title = Label(main, text='Video-Based Abnormal Driving Behavior Detection via Deep Learning Fusions',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
loadButton = Button(main, text="Generate & Load AWGRD Model", command=loadModel)
loadButton.place(x=50,y=200)
loadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=250)


uploadButton = Button(main, text="Upload Video", command=upload)
uploadButton.place(x=50,y=300)
uploadButton.config(font=font1)

uploadButton = Button(main, text="Start Behaviour Monitoring", command=startMonitoring)
uploadButton.place(x=50,y=350)
uploadButton.config(font=font1)

# exitButton = Button(main, text="Exit", command=exit)
# exitButton.place(x=50,y=400)
# exitButton.config(font=font1)

exitButton = Button(main, text="live cam", command=livecam)
exitButton.place(x=50,y=450)
exitButton.config(font=font1)

main.config(bg='chocolate1')
main.mainloop()'''


if __name__ == '__main__':

	app.run(debug=True)
