#!/usr/bin/env python
# coding: utf-8

# In[64]:


# Importing Libraries


# In[65]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# dropout is used for selecting user defined percentage of random neurons from hidden layers for calculation for more accuracy
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# In[66]:


# Reading Data


# In[67]:


d_1 = pd.read_csv('G:/Internship/Data set/2nd day/data.csv')
d_1.head()


# In[68]:


# Splitting the data 


# In[69]:


x = d_1.iloc[:,1:-1].values
y = d_1.iloc[:,-1:].values
print(x,y)


# In[70]:


# train_test_split


# In[71]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)


# In[72]:


# Model


# In[73]:


model = Sequential()
model.add(LSTM(64,return_sequences=True, input_shape=(11,1)))
model.add(Dropout(0.5))
model.add(LSTM(20,return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')


# In[74]:


model.fit(x_train, y_train, batch_size=11, epochs=200, verbose=0 )


# In[75]:


# Predicted


# In[76]:


y_pred = model.predict(x_test)


# In[77]:


print(y_pred)


# In[78]:


print(y_test)


# In[80]:


# print(accuracy_score(y_test, y_pred))
Mean_squared_error = sum((abs(y_test-y_pred))**2)/len(y_test)
print(Mean_squared_error)


# In[ ]:




