#!/usr/bin/env python
# coding: utf-8

# # Import

# In[69]:


import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# # Data

# In[70]:


rng = np.random.RandomState(1)
x = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(x[:,0], x[:, 1])
plt.axis('equal')


# # Model

# In[71]:


pca = PCA(n_components=2)
pca.fit(x)


# # Print

# In[72]:


# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

print(pca.explained_variance_)


# # Function defined

# In[73]:


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',linewidth=2, shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


# # Plot data

# In[76]:


plt.scatter(x[:,0], x[:,1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');


# In[ ]:




