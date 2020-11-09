# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 22:33:51 2020

@author: 현
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:16:35 2020

@author: 현
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from konlpy.tag import Okt
import json
import os
from pprint import pprint

data=pd.read_csv('onlybinary_test.csv', sep=",")

#print(data.head()) 출력확인
##print(len(data)) 크기확인
##print(type(data)) 타입확인


test_data = pd.DataFrame(data, columns=['발화','감정'])
print(test_data.head()) #Sentence랑 Emotion잘 뽑았는지 확인


after_data = []

for index in range(len(test_data)):
    #print(train_data['Sentence'][index])
    #print(train_data['Emotion'])
    okt = Okt()
    if index%50 ==0:
        print(index)
        
    def tokenize(doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)] 

    #pprint(tokenize(train_data['Sentence'][index]))
    
    tokens= tokenize(test_data['발화'][index])    
    feeling=test_data['감정'][index]
    feeling=int (feeling)
    
    after_data = after_data + [[tokens,feeling]]
    #pprint(after_data) 
    #print(type(after_data))

'''
#딕셔너리, 이뿌게 보이기
after_data = []
for index in range(10):
    #print(train_data['Sentence'][index])
    #print(train_data['Emotion'])
    okt = Okt()
    if index%50 ==0:
        print(index)
        
    def tokenize(doc):
        # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)] 

    #pprint(tokenize(train_data['Sentence'][index]))
    
    tokens= tokenize(train_data['Sentence'][index])    
    feeling=train_data['Emotion'][index]
    feeling=int(feeling)
    
    data_dict =dict()
    data_dict.update(zip(['Token','Emotion'],[tokens,feeling]))
    after_data.append(data_dict)
    #pprint(after_data) 
    #print(type(after_data))
'''


with open('test_data_binary.json', 'w', encoding="utf-8") as make_file:
    json.dump(after_data, make_file, ensure_ascii=False, indent="\t")
    print('done')

pprint(after_data)

