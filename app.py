from flask import Flask, redirect, send_file, url_for, render_template, request, Response
from werkzeug.utils import secure_filename
import face_recognition as fr
import numpy as np
import os
import cv2
from datetime import datetime
import random
import string 
import pandas as pd

app = Flask(__name__)
global UPLOAD_FOLDER
global UPLOAD_ATTENDANCE_FOLDER
random_key = ''.join(random.choice(string.ascii_letters) for i in range(8))
UPLOAD_FOLDER = 'image_folders/' 
IMAGE_FOLDER = UPLOAD_FOLDER + random_key
TXT_FILE = random_key + ".txt"

if os.path.exists(IMAGE_FOLDER):
    new_random_key = ''.join(random.choice(string.ascii_letters) for i in range(8)) 
    IMAGE_FOLDER = UPLOAD_FOLDER + new_random_key
    TXT_FILE = new_random_key + ".txt"


@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/submit.html')
def submit():
    return render_template("submit.html")

@app.route('/submit2.html')
def submit2():
    return render_template("submit2.html")

@app.route('/instructions.html')
def dello():
    return render_template("instructions.html")

@app.route('/submit.html', methods=['POST'])
def upload_file():
    if(request.method == "POST"):
        if('imagefolder' in request.files):
            os.mkdir(IMAGE_FOLDER)
            uploaded_files = request.files.getlist('imagefolder')
            for file in uploaded_files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(IMAGE_FOLDER, filename))
    return redirect(url_for('submit'))

@app.route('/download')
def download():
    path = os.path.join("attendance", TXT_FILE)
    with open(os.path.join("attendance", TXT_FILE),'r+') as f:
        f.write("\n----------------------\n ABSENT:")
        listOfNames = []
        for line in f.readlines():
            enter = line.split(',')
            listOfNames.append(enter[0])
        for name in os.listdir(IMAGE_FOLDER):
            if name[:-4] not in listOfNames:
                f.write(f'\n{name[:-4]}')

    return send_file(path, as_attachment=True)

        

def createFile():
    if not os.path.exists(os.path.join("attendance", TXT_FILE)):
        with open(os.path.join("attendance", TXT_FILE), 'w') as file:
            file.write('ATTENDANCE FOR ' + str(datetime.now().strftime('%m/%d/%Y')) + "\n")

def Attendance(name):
    #create file
    with open(os.path.join("attendance", TXT_FILE),'r+') as f:
        listOfNames = []
        for line in f.readlines():
            enter = line.split(',')
            listOfNames.append(enter[0])
        if name not in listOfNames:
            date = datetime.now().strftime('%H:%M:%S')
            f.write(f'\n{name},{date}')

def startRecognition():
    #0 for starting 
    if os.path.exists(IMAGE_FOLDER):
        images_path = os.listdir(IMAGE_FOLDER)
        names = [(os.path.splitext(image_path)[0]) for image_path in images_path]
        images = [cv2.imread(IMAGE_FOLDER + "/" + img) for img in images_path]
        known_face_encodings = [(fr.face_encodings(image)[0]) for image in images]
    global cap
    cap = cv2.VideoCapture(0)
    if len(os.listdir(IMAGE_FOLDER)) != 0:
        while True: 
                #ret returns True or False if webcam is on
                ret, image = cap.read()
                newImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                live_face_location = fr.face_locations(newImage)
                live_face_encoding = fr.face_encodings(newImage)

                for faceEncode, faceLocation in zip(live_face_encoding, live_face_location):
                    compare_face_encodings = fr.compare_faces(known_face_encodings, faceEncode)
                    distance_between_faces = fr.face_distance(known_face_encodings, faceEncode)
                    matchIdx = np.argmin(distance_between_faces) 
                    if(compare_face_encodings[matchIdx]):
                        name = names[matchIdx]
                        top, right, bottom, left = faceLocation
                        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), thickness=2)
                        (width, height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, thickness=1)
                        cv2.rectangle(image, (left, top - 25), (left + width, top), (255, 0, 0), thickness=-1)
                        cv2.putText(image, name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color=(255, 255, 255), thickness=1)
                        createFile()
                        Attendance(name)
                r, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
        
        

@app.route('/video_feed')
def video_feed():
    return Response(startRecognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

