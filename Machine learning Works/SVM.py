#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[191]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# # Read Data

# In[192]:


d_1 = pd.read_csv('G:/Internship/Data set/2nd day/data.csv')
d_1.head()


# In[193]:


x = d_1.iloc[:,1:-1].values
y = d_1.iloc[:,-1:].values
x,y


# # Train_Test_Split

# In[194]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# # Model

# In[195]:


# svclassifier = SVC(kernel ='poly', degree=12).fit(x_train, y_train)
svclassifier = SVC(kernel ='linear').fit(x_train, y_train)
# svclassifier = SVC(kernel='rbf').fit(x_train, y_train)
# svclassifier = SVC(kernel='sigmoid').fit(x_train, y_train)


# # Output

# In[196]:


y_pred = svclassifier.predict(x_test)


# In[197]:


print(y_pred)


# In[198]:


print(y_test)


# In[199]:


print(accuracy_score(y_test, y_pred))

