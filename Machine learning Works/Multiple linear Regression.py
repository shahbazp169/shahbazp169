#!/usr/bin/env python
# coding: utf-8

# In[200]:


# Import packages
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

#load the data
#define x and y
#Create a multi-output regressor
data=pd.read_excel('G:/Internship/Data set/DNN_dataset.xlsx')


# In[201]:


data


# In[202]:


x = data.iloc[:,:-2]
y = data.iloc[:,-2:]

x


# In[203]:


y


# In[204]:


x, y = make_regression(n_targets=2)


# In[205]:


x


# In[206]:


y


# In[207]:


#showing the shape


# In[208]:



x.shape


# In[209]:


y.shape


# In[210]:


#Split data into train and test


# In[211]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# In[212]:


#Model building


# In[213]:


clf = MultiOutputRegressor(RandomForestRegressor(max_depth=2, random_state=0))


# In[214]:


# Prediction and scoring


# In[215]:


clf.fit(x_train, y_train)


# In[216]:


clf.predict(x_test[[0]])


# In[217]:


clf.score(x_test, y_test, sample_weight=.one)

