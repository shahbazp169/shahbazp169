#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
x = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
y = np.array([1,1,1,2,2,2])
clf = LinearDiscriminantAnalysis()
clf.fit(x,y)


# In[4]:


#LinearDiscriminantAnalysis


# In[5]:


print(x)
print(y)
print(clf.predict([[-0.8, -1]]))

