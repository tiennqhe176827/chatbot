import nltk
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import re
import unicodedata

with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

ds_tu = []
ds_nhan = []
documents = []


for d in data['intents']:
    for p in d['patterns']:
        n = ''.join(c for c in unicodedata.normalize('NFD', p) if unicodedata.category(c) != 'Mn')
        n = re.sub(r'[^\w\s]', ' ', n.lower())
        w = nltk.word_tokenize(n)
        ds_tu.extend(w)
        documents.append((w, d['tag']))
        if d['tag'] not in ds_nhan:
            ds_nhan.append(d['tag'])

ds_tu = sorted(list(set(ds_tu)))
ds_nhan = sorted(list(set(ds_nhan)))

print(len(documents), "số cặp câu-nhãn")
print(len(ds_nhan), "nhãn:", ds_nhan)
print(len(ds_tu), "từ:", ds_tu)

with open('texts.pkl', 'wb') as file:
    pickle.dump(ds_tu, file)
with open('labels.pkl', 'wb') as file:
    pickle.dump(ds_nhan, file)

training = []

for doc in documents:
    bag = []
    xl_doc = doc[0]
    xl_doc = [word.lower() for word in xl_doc]
    for w in ds_tu:
        bag.append(1) if w in xl_doc else bag.append(0)
    
    output_row = [0] * len(ds_nhan)
    output_row[ds_nhan.index(doc[1])] = 1 
    training.append([bag, output_row])

random.shuffle(training)

train_x = [x[0] for x in training]
train_y = [x[1] for x in training]

train_x = np.array(train_x)
train_y = np.array(train_y)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=300, batch_size=5, verbose=1)

model.save('model.keras')

print("Tạo Model thành công!")
