#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Importing Dataset

# In[17]:


dataset = pd.read_csv('G:/Internship/Data set/K_Means_Clustering/Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values
dataset


# # Using the elbow method 

# In[18]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++' , random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[19]:


# Plotting Graphs


# In[20]:


plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[21]:


# Training the K-Means model on the dataset


# In[22]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)
y_kmeans


# In[23]:


# Visualising the clusters


# In[24]:


plt.scatter(x[:,0], x[:,1], c=y_kmeans, cmap = 'gist_rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s=300, c = 'yellow')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')


# In[ ]:




