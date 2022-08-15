#!/usr/bin/env python
# coding: utf-8

# # Importing The Libraries

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# # Reading the data

# In[6]:


d_1=pd.read_csv('G:/Internship/Data set/2nd day/data.csv')
d_1.head()


# # Selecting dataset for training

# In[7]:


x=d_1.iloc[:,:]
train_dataset = x.sample(frac=0.7)
print(train_dataset)


# # Defining x train, y train, x test and y test

# In[8]:


x_tr=train_dataset.iloc[:,:-1]
print(x_tr)
y_tr=train_dataset.iloc[:,-1:]
print(y_tr)
test_dataset = x.drop(train_dataset.index)
x_ts = test_dataset.iloc[:,:-1]
print(x_ts)
y_ts = test_dataset.iloc[:,-1:]
print(y_ts)


# # Model 

# In[9]:


(x_train,y_train),(x_test,y_test)=(x_tr,y_tr),(x_ts,y_ts)


# In[10]:


model_dt = RandomForestClassifier().fit(x_train, y_train)


# # Output

# In[11]:


y_pred= model_dt.predict(x_test)


# # Predicted

# In[12]:


print(y_pred)


# # Test

# In[13]:


print(y_test)


# # Accuracy

# In[14]:


print(100*accuracy_score(y_test,y_pred))

