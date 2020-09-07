# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:21:45 2020

@author: 현
"""
import pandas as pd

sr = pd.Series([17000, 18000, 1000,5000],index=["피자","치킨","콜라","맥주"])
print(sr)
print(sr.values)
print(sr.index)