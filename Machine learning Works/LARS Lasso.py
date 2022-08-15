#!/usr/bin/env python
# coding: utf-8

# # Importing The Libraries

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


# # Reading the data

# In[28]:


d_1=pd.read_csv('G:/Internship/Data set/DNN_dataset.csv')
d_1.head()


# # Selecting dataset for training

# In[29]:


x=d_1.iloc[:,:]
train_dataset = x.sample(frac=0.7)
print(train_dataset)


# # Defining x train, y train, x test and y test

# In[30]:


x_tr=train_dataset.iloc[:,:-2]
print(x_tr)
y_tr=train_dataset.iloc[:,-2:]
print(y_tr)
test_dataset = x.drop(train_dataset.index)
x_ts = test_dataset.iloc[:,:-2]
print(x_ts)
y_ts = test_dataset.iloc[:,-2:]
print(y_ts)


# # Model 

# In[31]:


(x_train,y_train),(x_test,y_test)=(x_tr,y_tr),(x_ts,y_ts)


# In[32]:


model = linear_model.LassoLars(alpha=.1, normalize=False)


# In[33]:


model.fit(x_train, y_train)


# In[34]:


# model_dt = linear_model.Ridge(alpha=.5).fit(x_train, y_train)


# # Output

# In[35]:


y_pred= model.predict(x_test)


# # Predicted

# In[36]:


print(y_pred)


# # Test

# In[37]:


print(y_test)


# # Accuracy

# In[38]:


# print(accuracy_score(y_test,y_pred))


# In[39]:


mse=mean_squared_error(y_test, y_pred)
print("MSE=",mse)

rmse=np.sqrt(mse)
print("RMSE=",rmse)
MAE= mean_absolute_error(y_test, y_pred)
print("MAE: ",MAE)
r2=r2_score(y_test, y_pred)
print("R2 score=",r2)
MAPE= mean_absolute_percentage_error(y_test, y_pred)
print("MAPE: ",MAPE)

