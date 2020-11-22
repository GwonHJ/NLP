# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:02:51 2020

@author: 현
"""
import pandas as pd
import matplotlib.pyplot as plt
import konlpy
from konlpy.tag import Okt 
from keras.preprocessing.text import Tokenizer 
import numpy as np




train_data = pd.read_csv("합본.csv")

##부정 중립 긍정 수 보기
print(train_data.groupby('label').size().reset_index(name='count')) 

okt = Okt()
#%%sentence전처리

print(1)


stopwords = ['의', '가', '이', '은', '들', '는', '좀', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

X_train = [] 
for sentence in train_data['Sentence']: 
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_train.append(temp_X) 

max_words = 35000
tokenizer = Tokenizer(num_words = max_words) 
tokenizer.fit_on_texts(X_train) 
X_train = tokenizer.texts_to_sequences(X_train) 

#%%


print(2)
print("문장의 최대 길이 : ", max(len(l) for l in X_train)) 
print("문장의 평균 길이 : ", sum(map(len, X_train))/ len(X_train)) 
'''
그래프
plt.hist([len(s) for s in X_train], bins=50) 
plt.xlabel('length of Data') 
plt.ylabel('number of Data') 
plt.show()
'''
#%%

print(3)
y_train = []

##y_train원-핫 인코딩
for i in range(len(train_data['label'])): 
    if train_data['label'].iloc[i] == 1: 
        y_train.append([0, 0, 1]) 
    elif train_data['label'].iloc[i] == 0:
        y_train.append([0, 1, 0]) 
    elif train_data['label'].iloc[i] == -1:
        y_train.append([1, 0, 0])

y_train = np.array(y_train)



#%%데이터셋 나누기
from sklearn.model_selection import train_test_split


print(4)
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3)


#%%훈련

from keras.layers import Embedding, Dense, LSTM 
from keras.models import Sequential 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences 


print(5)
max_len = 20 # 전체 데이터의 길이를 20로 맞춘다 

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen = max_len)
'''
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
'''

print(6)
model = Sequential() 
model.add(Embedding(max_words, 128)) 
model.add(LSTM(128)) 
model.add(Dense(3, activation='softmax')) 

##정확도가 올랐을때만 모델을 저장, 과적합이 일어날거같으면 중지
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model2.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) 
history = model.fit(x_train, y_train, batch_size=500, epochs=30, callbacks=[es, mc], validation_data=(x_test, y_test))
