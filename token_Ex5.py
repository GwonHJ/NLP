# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 02:36:30 2020

@author: 현
"""
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff, he dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
print(sent_tokenize(text))