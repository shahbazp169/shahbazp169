#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
# from tensorflow.keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# dropout is used for selecting user defined percentage of random neurons from hidden layers for calculation for more accuracy
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Reading Data


# In[4]:


d_1 = pd.read_csv('G:/Internship/Data set/2nd day/data.csv')
d_1.head()


# In[5]:


# Splitting the data 


# In[6]:


x = d_1.iloc[:,1:-1].values
y = d_1.iloc[:,-1:].values
print(x,y)


# In[7]:


# train_test_split


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=0)


# In[9]:


# Model


# In[10]:


model = Sequential()
model.add(Conv1D(32,2, activation='relu', input_shape=(11,1)))
# 32 is the filter size and 2 is the kernal size
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse')
model.summary()


# In[11]:


model.fit(x_train, y_train, batch_size=11, epochs=200, verbose=0)


# In[12]:


# Predicted


# In[13]:


y_pred = model.predict(x_test)
print(y_pred)
print(y_test)


# In[14]:


# print(accuracy_score(y_test, y_pred))
Mean_squared_error = sum((abs(y_test-y_pred))**2)/len(y_test)
print(Mean_squared_error)


# In[ ]:




