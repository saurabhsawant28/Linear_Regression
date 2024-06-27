#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('homeprices.csv')
df


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color ='red',marker = '+')
plt.show()


# In[8]:


new_df = df.drop('price',axis = 'columns')
new_df


# In[9]:


price = df.price
price


# In[10]:


# create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)


# In[11]:


# Predict price of home with area = 3300 sqr ft
reg.predict([[3300]])


# In[12]:


reg.coef_


# In[13]:


reg.intercept_


# In[14]:


# Y = m*X + b ( m is coefficient and b is intercept)
3300*135.78767123 + 180616.43835616432


# In[17]:


# Predict price of home with area = 5000 sqr ft
reg.predict([[5000]])


# In[18]:


# Generate CSV file with list of home price conditions


# In[22]:


area_df = pd.read_csv("areas.csv")
area_df.head(3)


# In[23]:


p = reg.predict(area_df)
p


# In[24]:


area_df.to_csv("prediction.csv")


# In[ ]:




