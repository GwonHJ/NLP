# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 01:58:41 2020

@author: 현
"""
from gensim.models import KeyedVectors


loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드

model_result = loaded_model.most_similar("man")
print(model_result)
