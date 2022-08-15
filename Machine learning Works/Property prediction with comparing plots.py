#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# ## Models

# In[39]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline


# ## Preprocessing

# In[40]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# ## Results

# In[41]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


# # Reading Dataset

# In[42]:


data=pd.read_excel('G:/Internship/Ammus sample/RAMAN.xlsx', sheet_name='Day 28 !0')               
data.head()


# # Defining Variables

# In[43]:


# data = data1.copy()

# # apply normalization techniques
# for column in data.columns:
# 	data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())	

# # view normalized data
# print(data)


x = data.iloc[:,:-1].values
y = data.iloc[:,-1:].values
print(x,y)


# # train_test_split

# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)


# # Models

# ## Linear Model Ridge

# In[45]:


model1 = linear_model.Ridge(alpha=0.1)


# In[46]:


model1.fit(x_train, y_train)


# In[47]:


y1_pred= model1.predict(x_test)
print(y1_pred)


# In[48]:


mse=mean_squared_error(y_test, y1_pred)
print("MSE=",mse)
rmse=np.sqrt(mse)
print("RMSE=",rmse)
MAE= mean_absolute_error(y_test, y1_pred)
print("MAE: ",MAE)
r2=r2_score(y_test, y1_pred)
print("R2 score=",r2)
MAPE= mean_absolute_percentage_error(y_test, y1_pred)
print("MAPE: ",MAPE)


# ## Kernel Ridge

# In[49]:


model2 = KernelRidge(alpha=1.0)


# In[50]:


model2.fit(x_train, y_train)


# In[51]:


y2_pred= model2.predict(x_test)
print(y2_pred)


# In[52]:


mse=mean_squared_error(y_test, y2_pred)
print("MSE=",mse)
rmse=np.sqrt(mse)
print("RMSE=",rmse)
MAE= mean_absolute_error(y_test, y2_pred)
print("MAE: ",MAE)
r2=r2_score(y_test, y2_pred)
print("R2 score=",r2)
MAPE= mean_absolute_percentage_error(y_test, y2_pred)
print("MAPE: ",MAPE)


# ## Bayesian Ridge

# In[53]:


model3 = linear_model.BayesianRidge()


# In[54]:


model3.fit(x_train, y_train)


# In[55]:


y3_pred= model3.predict(x_test)
# print(y3_pred)
a = y3_pred
print(a)


# In[56]:


mse=mean_squared_error(y_test, y3_pred)
print("MSE=",mse)
rmse=np.sqrt(mse)
print("RMSE=",rmse)
MAE= mean_absolute_error(y_test, y3_pred)
print("MAE: ",MAE)
r2=r2_score(y_test, y3_pred)
print("R2 score=",r2)
MAPE= mean_absolute_percentage_error(y_test, y3_pred)
print("MAPE: ",MAPE)


# 

# In[57]:


# model4 = linear_model.Lasso(alpha=0.1)


# In[58]:


# model4.fit(x_train, y_train)


# In[59]:


# y4_pred= model4.predict(x_test)


# In[60]:


# mse=mean_squared_error(y_test, y4_pred)
# print("MSE=",mse)
# rmse=np.sqrt(mse)
# print("RMSE=",rmse)
# MAE= mean_absolute_error(y_test, y4_pred)
# print("MAE: ",MAE)
# r2=r2_score(y_test, y4_pred)
# print("R2 score=",r2)
# MAPE= mean_absolute_percentage_error(y_test, y4_pred)
# print("MAPE: ",MAPE)


# 

# In[61]:


# model5 = linear_model.LassoLars(alpha=.1, normalize=False)


# In[62]:


# model5.fit(x_train, y_train)


# In[63]:


# y5_pred= model5.predict(x_test)


# In[64]:


# mse=mean_squared_error(y_test, y5_pred)
# print("MSE=",mse)
# rmse=np.sqrt(mse)
# print("RMSE=",rmse)
# MAE= mean_absolute_error(y_test, y5_pred)
# print("MAE: ",MAE)
# r2=r2_score(y_test, y5_pred)
# print("R2 score=",r2)
# MAPE= mean_absolute_percentage_error(y_test, y5_pred)
# print("MAPE: ",MAPE)


# 

# In[65]:


# model6 = model = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])


# In[66]:


# model6.fit(x_train, y_train)


# In[67]:


# y6_pred= model6.predict(x_test)


# In[68]:


# mse=mean_squared_error(y_test, y6_pred)
# print("MSE=",mse)
# rmse=np.sqrt(mse)
# print("RMSE=",rmse)
# MAE= mean_absolute_error(y_test, y6_pred)
# print("MAE: ",MAE)
# r2=r2_score(y_test, y6_pred)
# print("R2 score=",r2)
# MAPE= mean_absolute_percentage_error(y_test, y6_pred)
# print("MAPE: ",MAPE)


# 

# In[69]:


# model7 = tree.DecisionTreeRegressor()


# In[70]:


# model7.fit(x_train, y_train)


# In[71]:


# y7_pred= model.predict(x_test)


# In[72]:


# Denormalizing

# y_test = y_test*(data['CS(Exp.)'].max() - data['CS(Exp.)'].min())/data['CS(Exp.)'].min()
# y1_pred = y1_pred*(data1.iloc[:,-1:].max().values-data1.iloc[:,-1:].min().values)/data1.iloc[:,-1:].min().values
# y2_pred = y2_pred*(data1.iloc[:,-1:].max().values-data1.iloc[:,-1:].min().values)/data1.iloc[:,-1:].min().values
# y3_pred = y3_pred*(data1.iloc[:,-1:].max().values-data1.iloc[:,-1:].min().values)/data1.iloc[:,-1:].min().values
# y_test = y_test*(data1.iloc[:,-1:].max().values-data1.iloc[:,-1:].min().values)/data1.iloc[:,-1:].min().values
# print(y_test)
# print(data1.iloc[:,-1:].max().values)
# print(data1.iloc[:,-1:].min().values)
# print(y_test)


# # Plotting Test v/s Predict

# In[73]:


# plt.plot(x_test, marker='o', linestyle='-', label='Actual load')
plt.plot(y_test, marker='o', linestyle='-', label='Actual value')
plt.plot(y1_pred, marker='o', linestyle='-', label='Ridge')
plt.plot(y2_pred, marker='o', linestyle='-', label='KRR')
plt.plot(y3_pred, marker='o', linestyle='-', label='Bayesian')
# plt.plot(y4_pred, marker='o', linestyle='-', label='Lasso')
# plt.plot(y5_pred, marker='o', linestyle='-', label='LARS Lasso')
# plt.plot(y6_pred, marker='o', linestyle='-', label='Pipeline')
# plt.plot(y7_pred, marker='o', linestyle='-', label='DTR')
#plt.plot(y_predict_2,label='Future')
#plt.plot(x_test11, marker='o', linestyle='-',label='Train')
plt.xlim([0,90])
plt.ylim([0,100])
plt.grid()
plt.legend(loc=1, prop={'size': 8}, numpoints = 1)
#plt.title('Loss function')
# plt.title('Actual v/s Predicted', fontname="Times New Roman",fontweight="bold", fontsize=20)
plt.xlabel('No. of test samples', fontsize=12, fontname="Times New Roman")
plt.ylabel('Compressive Strength (MPa)', fontsize=12, fontname="Times New Roman")
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.savefig('Actual vs Predict.png', format='png', dpi=600)
plt.show()


# In[ ]:





# In[ ]:




