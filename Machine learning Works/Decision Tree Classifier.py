#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classifier

# # Importing the libraries

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


# # Reading the data

# In[30]:


d_1=pd.read_csv('G:/Internship/Data set/2nd day/data.csv')
d_1.head()


# # Selecting the dataset  for training

# In[31]:


x=d_1.iloc[:,:]
train_dataset = x.sample(frac=0.7)
print(train_dataset)


# # Defining X and Y

# In[32]:


x_tr=train_dataset.iloc[:,:-1]
print(x_tr)
y_tr=train_dataset.iloc[:,-1:]
print(y_tr)


# # Now test dataset and dividing

# In[33]:


test_dataset = x.drop(train_dataset.index)
x_ts = test_dataset.iloc[:,:-1]
print(x_ts)
y_ts = test_dataset.iloc[:,-1:]
print(y_ts)


# # Model

# In[34]:


(x_train,y_train),(x_test,y_test)=(x_tr,y_tr),(x_ts,y_ts)


# In[35]:


model_dt = DecisionTreeClassifier().fit(x_train, y_train)


# # Output

# In[36]:


y_pred= model_dt.predict(x_test)


# # Predicted

# In[37]:


print(y_pred)


# # Test

# In[38]:


print(y_test)


# # Accuracy

# In[40]:


print(100*accuracy_score(y_test,y_pred))

