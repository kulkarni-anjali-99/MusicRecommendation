from flask import Flask,request
from flask_cors import CORS,cross_origin
from utility import *

import json
import os

app = Flask(__name__)
cors = CORS(app, resources={r"/react/*": {"origins": "https://react-emotion.herokuapp.com/"}})
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/getOnlyEmotion',methods=['POST','GET'])
@cross_origin()
def get_Only_Emotion():
    if request.method == 'POST':
        print("\n\n")
        print("......................................................GetOnlyEmotion Called............................................................")
        f = request.files['file']
        print("\n\n")
        print("......................................................Audio File Recieved..........................................................")
        f.save(os.path.join(basedir, f.filename))
        print("......................................................Audio File Saved.............................................................")
        try:
            os.remove("file-c.wav")
        except OSError:
            pass
        print("\n\n")
        print("......................................................Audio File Converting........................................................")
        os.system("ffmpeg -i {0} file-c.wav".format(f.filename))
        print("......................................................Audio File Converted to wav format...........................................")
        print("\n\n")
        print("......................................................Audio File Unsilencing.......................................................")
        os.system("unsilence file-c.wav file-c.wav -ao -y")
        print("......................................................Audio File Unsilened.........................................................")
        emotion = predict_emotion()
        response = {"emotion":emotion}
        print("\n\nResponse is :\n")
        print(response)
        print("......................................................getOnlyEmotion Call ended........................................................")
        print("\n\n")
        return response

@app.route('/react/getEmotion',methods=['POST','GET'])
@cross_origin()
def get_Emotion():
    if request.method == 'POST':
        print("\n\n")
        print("......................................................GetEmotion Called............................................................")
        f = request.files['file']
        print("\n\n")
        print("......................................................Audio File Recieved..........................................................")
        f.save(os.path.join(basedir, f.filename))
        print("......................................................Audio File Saved.............................................................")
        try:
            os.remove("file-c.wav")
        except OSError:
            pass
        print("\n\n")
        print("......................................................Audio File Converting........................................................")
        os.system("ffmpeg -i {0} file-c.wav".format(f.filename))
        print("......................................................Audio File Converted to wav format...........................................")
        print("\n\n")
        print("......................................................Audio File Unsilencing.......................................................")
        os.system("unsilence file-c.wav file-c.wav -ao -y")
        print("......................................................Audio File Unsilened.........................................................")
        data = request.form.get('songFeatures', '')
        data = json.loads(data)
        emotion = predict_emotion()
        recommendation = recommendations(data,emotion)
        response = {"emotion":emotion,"recommendation":recommendation}
        print("\n\nResponse is :\n")
        print(response)
        print("......................................................getEmotion Call ended........................................................")
        print("\n\n")
        print(request.form.keys())
        return response

@app.route('/react/getRecommendation',methods=['POST','GET'])
@cross_origin()
def get_recommendation():
    if request.method == 'POST':
        print("\n\n")
        print("......................................................getRecommendation Called.....................................................")
        emotion = request.form.get('emotion','')
        data = request.form.get('songFeatures', '')
        data = json.loads(data)
        recommendation = recommendations(data,emotion)
        response = {"emotion":emotion,"recommendation":recommendation}
        print("\n\nResponse is :\n")
        print(response)
        print("......................................................getRecommendation Call ended.................................................")
        print("\n\n")
        print(request.form.keys())
        print("Emotion received :",request.form["emotion"])
        return response

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True)

