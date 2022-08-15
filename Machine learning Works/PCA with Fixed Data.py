#!/usr/bin/env python
# coding: utf-8

# # Import

# In[63]:


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# # Data

# In[64]:


x = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])


# # Model

# In[65]:


pca = PCA(n_components=2)
pca.fit(x)


# # Print

# In[66]:


print(pca.explained_variance_ratio_)
print(pca.singular_values_)

print(pca.explained_variance_)


# # Function defined

# In[67]:


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# # Plot data

# In[68]:


plt.scatter(x[:,0], x[:,1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');

