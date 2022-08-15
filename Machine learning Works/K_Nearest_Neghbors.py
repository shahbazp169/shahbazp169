#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Importing Libraries


# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[19]:


# Importing Dataset
## Dataset about purchasing a car or not


# In[20]:


dataset = pd.read_csv('G:/Internship/Data set/K_Nearest_Neghbours/Social_Network_Ads.csv')
x =dataset.iloc[:,:-1].values
y =dataset.iloc[:,-1:].values
dataset


# In[21]:


# Splitting the Dataset into the training set and test set


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[23]:


# Feature scaling
## here mean and standard deviation has already used in x_train and we are going to use the same mean and standard deviation in x_test also, so only transform is needed


# In[24]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[25]:


# Training the K-NN model on the Training Dataset


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)


# In[27]:


# Predict a New Result


# In[30]:


print(classifier.predict(sc.transform([[30,87000]]))>0.5)


# In[ ]:




