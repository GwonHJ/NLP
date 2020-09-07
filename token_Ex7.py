# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 02:45:23 2020

@author: 현
"""
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))

x=word_tokenize(text)
pos_tag(x)

print(pos_tag(x))