#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import math


# In[2]:


df = pd.read_csv('G:/Internship/Data set/Linear/wind data')
df.head()


# In[3]:


#Traintest


# In[4]:


x = df.iloc[:,1:5].values
y = df.iloc[:,6].values
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[5]:


#Model


# In[6]:


degree = 3
reg = linear_model.LinearRegression()


# In[7]:


#Fit


# In[8]:


reg.fit(x_train, y_train)


# In[9]:


#Prediction
WS_Test = reg.predict(x_test)


# In[10]:


WS_Test = np.reshape(WS_Test,(-1,1))
y_test = np.reshape(y_test,(-1,1))


# In[11]:


#Performance matrix


# In[12]:


Mean_squared_error = sum((abs(y_test-WS_Test))**2)/len(y_test)

Root_mean_square = np.sqrt(sum((abs(y_test-WS_Test))**2)/len(y_test))

Average_error = sum(abs(y_test-WS_Test))/len(y_test)

Normalized_mean_square_error = sum((abs(y_test-WS_Test))**2)/sum((abs(y_test))**2)*100

Normalized_root_mean_square_error = (Root_mean_square)/(max(y_test))*100

weighted_mean_average_error = sum((abs(y_test-WS_Test))/sum(y_test))*100

Maximum_error = max(abs(y_test-WS_Test))


# In[13]:


print('MSE:=', Mean_squared_error,'\n', 'AE (MAE):=', Average_error, '\n', 'ME:=',Maximum_error, '\n', 'NMSE (%):=',Normalized_mean_square_error, '\n','NRMSE (%):=' , Normalized_root_mean_square_error,'\n','RMSE:=', Root_mean_square,'\n','WMAE:=', weighted_mean_average_error )


# In[14]:


print('Actual vs Prediction for last 24hrs testing data')
plt.plot(WS_Test, color='navy' , label='Regression')
plt.plot(y_test, color='red', label='Ground ruth')
plt.title('Actual vs Prediction')
plt.ylabel('Wind Power (KW)')
plt.xlabel('Time(hrs)')
plt.legend(loc="upper left")
plt.xlim(0,24)
plt.ylim(0,30000)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()


# In[15]:


#read in xls sheet


# In[16]:


List1 =[]
List1.append(Mean_squared_error)
List1.append(Root_mean_square)
List1.append(Average_error)
List1.append(Normalized_mean_square_error)
List1.append(Normalized_root_mean_square_error)
List1.append(weighted_mean_average_error)
List1.append(Maximum_error)
print(List1)


# In[17]:


#saving dataframe as csv file


# In[18]:


dff =pd.DataFrame(List1)
dff.to_csv('Error.csv')
print(dff)

