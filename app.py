import nltk
import json
import pickle
import numpy as np
import random
import speech_recognition as sr
from flask import Flask, render_template, request
import tensorflow as tf
import re
import unicodedata

ds_words = pickle.load(open('texts.pkl','rb'))
ds_nhan = pickle.load(open('labels.pkl','rb'))

model = tf.keras.models.load_model('model.keras', compile=False)

data = json.loads(open('data.json', encoding='utf-8').read())

def xuly_dulieu(text):
    n = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    dc=re.sub(r'[^\w\s]', ' ', n)
    xl_word = nltk.word_tokenize(dc)
    xl_word = [word.lower() for word in xl_word]
    return xl_word


def bow(input, words):
    xl_input = xuly_dulieu(input)
    bag = [0] * len(words)
    for s in xl_input:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def dudoan_nhan(input, model):
    p = bow(input, ds_words)
    res = model.predict(np.array([p]))[0]
    Nguong = 0.25
    kq_dudoan = []
    for i, r in enumerate(res):
        if r > Nguong:
            kq_dudoan.append([i, r])
    kq_dudoan.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in kq_dudoan:
        return_list.append({"nhan": ds_nhan[r[0]], "xacsuat": str(r[1])})
    return return_list


def get_phanhoi(ints, data_json):
    tag = ints[0]['nhan']
    yd = data_json['intents']
    for i in yd:
        if(i['tag']== tag):
            phanhoi = random.choice(i['responses'])
            break
    return phanhoi


def get_audio():
    mc = sr.Recognizer()
    with sr.Microphone() as source:
        audio = mc.listen(source, phrase_time_limit=5)
        try:
            text = mc.recognize_google(audio, language="vi-VN")
            return text
        except:
            return 0

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if userText == "mic":
        audio_text = get_audio()
        if audio_text:
            return audio_text
        else:
            return "0"
    else:
        ints = dudoan_nhan(userText, model)
        res = get_phanhoi(ints, data)
        return res


if __name__ == "__main__":
    app.run()
