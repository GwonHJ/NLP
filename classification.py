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



#수정_합본데이터 : 단발성+연속성 데이터 불러오기
train_data = pd.read_csv("수정_합본.csv")
#데이터가 부족한거 같아서 네이버 쇼핑리뷰 긍정 부정 데이터 추가했음
train_data2 = pd.read_csv("sample.csv")
#근데 데이터가 너무 많은거 같아서 그냥 10만까지만 불러옴
train_data2 = train_data2[:100000]
#아까 수정합본이랑 네이버리뷰데이터랑 합쳐줌
train_data = pd.concat([train_data,train_data2])

##부정 중립 긍정 수 보기 부정수가 너무 많아졌지만 일단 스킵...
print(train_data.groupby('Emotion').size().reset_index(name='count')) 

#%%sentence전처리 문장을 단어로 끊기 위한 전처리 한국어는 문장에 조사 이런거 때문에 영어랑 또 다른데 그거는 Okt를 불러서 얘가 알아서 처리하게 할거임.
#Okt말고도 한국어 전처리 하는게 있는데 일단 내가 따라한거는 Okt써서 Okt쓸거임
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

print(1)

okt = Okt()
#불용어 제거를 위해서 불용어 정의 불용어는 문장에서 제거해도 문장 뜻에 큰 영향을 안 미치는 애들 얘를 들면 나는 너를 사랑해에서 나 너 사랑해 이것만되도 문장 이해가 가능한거
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

X_train = []
cnt=-1 
for sentence in train_data['Sentence']: 
    cnt = cnt +1 
    if cnt%2000==0:
        print(cnt)
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화 이거 돌아간거 한번 보면 Okt가 알아서 문장을 단어화 시켜놓음. ex) 물먹은 -> 물/먹다 변형된 단어를 원형으로 바꿈
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_train.append(temp_X) 


#문자열은 훈련이 불가능 -> 배열로 만들어줌(원 핫 인코딩)
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

##y_train원-핫 인코딩 훈련시키기 위해서 원 핫 인코딩을 함
#딥러닝 훈련할때 원-핫 인코딩을 함 앞에 문장도 원 핫 인코딩을 함
for i in range(len(train_data['Emotion'])): 
    if train_data['Emotion'].iloc[i] == 1: 
        y_train.append([0, 0, 1]) 
    elif train_data['Emotion'].iloc[i] == 0:
        y_train.append([0, 1, 0]) 
    elif train_data['Emotion'].iloc[i] == -1:
        y_train.append([1, 0, 0])

y_train = np.array(y_train)



#%%데이터셋 나누기
from sklearn.model_selection import train_test_split

#트레인셋이랑 테스트셋 나누
print(4)
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1)


#%%훈련

from keras.layers import Embedding, Dense, LSTM, Dropout
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
#임베딩 : 원핫인코딩은 단어간의 관계성 파악이 불가능하고, 공간을 많이 차지해서 효율이 떨어짐 그래서 얘네를 실수형을 해서 배열의 크기를 줄이는 과정 이것도 임베딩이 알아서 해줌
#lstm이라는 모델을 쓸거임 안에 구조는 나도 잘 모름 그냥 막 씀
#덴스 3 감정이 3이니까 3개로 나오게 softmax는 3개별로 확률로 나오게 하는거 다 더하면 1임
#드롭아웃 : 과적합을 막게 하기 위해서 중간에 노드?같은거를 50프로 빼준다는거
#밑에 주석처리해놓은거는 내가 이제 뭐가 제일 정확도 높은지 확인하기 위해서 추가해준것들 여기 숫자랑 쌓는 수를 변경해가면서 어떤게 제일 정확도가 높게 나오는지 확인하고 있음
print(6)
model = Sequential()
model.add(Embedding(max_words,128)) 
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
'''
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
'''
model.add(LSTM(128, return_sequences = False)) 
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
'''
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
model.add(Dense(9, activation='relu')) 
model.add(Dropout(0.5)) # 드롭아웃 추가. 비율은 50%
'''
model.add(Dense(3, activation='softmax')) 
  
'''
##정확도가 올랐을때만 모델을 저장, 과적합이 일어날거같으면 중지
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model2.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
'''

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) 
history = model.fit(x_train, y_train, batch_size=500, epochs=30, validation_data=(x_test, y_test))

#로스, 정확도 변화 그래프
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'])  
plt.plot(epochs, history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%% 입력된 문장 예측값

from tensorflow.keras.models import load_model

loaded_model = load_model('best_model2.h5')

def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = loaded_model.predict(pad_new) # 예측
  print(score)
  
  
temp_str = input()
sentiment_predict(temp_str)
