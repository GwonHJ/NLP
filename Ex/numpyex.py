# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 21:36:22 2020

@author: 현
"""
import numpy as np

a = np.array([1,2,3,4,5])
print(type(a))
print(a)

b = np.array([[10, 20, 30], [ 60, 70, 80]]) 
print(b)

print(b.ndim) 
print(b.shape)

print(a.ndim)
print(a.shape)

a = np.zeros((2,3))
print(a)

a = np.ones((2,3))
print(a)

a = np.full((2,2), 7)
print(a)

a = np.eye(3)
print(a)

a = np.random.random((2,2))
print(a)


a = np.arange(10)
print(a)

a = np.arange(1, 10, 2)
print(a)

a = np.array(np.arange(30)).reshape((5,6))
print(a)

a = np.array([[1, 2, 3], [4, 5, 6]])
b=a[0:2, 0:2]
print(b)

b=a[0, :]
print(b)

b=a[:, 1]
print(b)

a = np.array([[1,2], [4,5], [7,8]])
b = a[[2, 1],[1, 0]]
print(b)

x = np.array([1,2,3])
y = np.array([4,5,6])


b = x + y
print(b)

b = x - y
print(b)

b = b * x
print(b)

b = b / x
print(b)

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

c = np.dot(a, b)
print(c)
