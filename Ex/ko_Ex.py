# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 02:48:38 2020

@author: 현
"""
from konlpy.tag import Okt 
from nltk.tag import pos_tag
 
okt=Okt()  
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  

print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
