# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 20:41:26 2020

@author: 현
"""
import numpy as np
from konlpy.tag import Okt
import json
import os
from pprint import pprint
import nltk
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

okt = Okt()


with open('train_data_binary.json',"r",encoding="utf-8") as f:
    train_docs = json.load(f)
with open('test_data_binary.json',"r",encoding="utf-8") as f:
    test_docs = json.load(f)
#pprint(train_docs[0]) ##불러온거 확인

tokens = [t for d in train_docs for t in d[0]]
#print(len(tokens))

text = nltk.Text(tokens, name='NMSC')

# 전체 토큰의 개수
#print(len(text.tokens))

# 중복을 제외한 토큰의 개수
#print(len(set(text.tokens)))            

# 출현 빈도가 높은 상위 토큰 10개
#pprint(text.vocab().most_common(10))

'''
font_fname = 'c:/windows/fonts/gulim.ttc'
font_name = font_manager.FontProperties(fname=font_fname).get_name()
rc('font', family=font_name)


plt.figure(figsize=(20,10))
text.plot(50)
'''

#너무 오래걸리면 빈도수 많은 상위 단어 개수 줄이기
selected_words = [f[0] for f in text.vocab().most_common(5000)]
#selected_words = [f[0] for f in text.vocab().most_common(5000)]

print("d")

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
print("dd")
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]

print("ddd")

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')

'''


num_classes = 7

max_len = 20


x_train = pad_sequences(x_train, maxlen=max_len) # 훈련용 뉴스 기사 패딩
x_test = pad_sequences(x_test, maxlen=max_len) # 테스트용 뉴스 기사 패딩


y_train = to_categorical(y_train) # 훈련용 뉴스 기사 레이블의 원-핫 인코딩
y_test = to_categorical(y_test) # 테스트용 뉴스 기사 레이블의 원-핫 인코딩

model = Sequential()
model.add(Embedding(5000, 200))
model.add(LSTM(200))
model.add(Dense(7, activation='softmax'))


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



history = model.fit(x_train, y_train, batch_size=32, epochs=30, callbacks=[es, mc], validation_data=(x_test, y_test))


loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))



'''

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(5000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

model.fit(x_train, y_train, epochs=10, batch_size=64)
results = model.evaluate(x_test, y_test)
  
