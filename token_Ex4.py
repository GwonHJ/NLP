# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 02:33:44 2020

@author: 현
"""
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))