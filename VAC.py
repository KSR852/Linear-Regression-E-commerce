#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv('Ecommerce Customers')
df.shape


# In[7]:


df.info()


# In[8]:


df.isna().sum()


# In[9]:


df.head()


# In[10]:


df.drop(['Email','Address'],axis=1,inplace=True)
df.head()


# In[11]:


df.corr()


# In[12]:


df['Avatar'].unique()


# In[13]:


plt.figure(figsize=(10,10))
sns.pairplot(data=df)
plt.show()


# In[14]:


df.columns


# In[15]:


cols_to_scale = ['Avg. Session Length', 'Time on App', 'Time on Website','Length of Membership']
scale = MinMaxScaler()
scalled = scale.fit_transform(df[cols_to_scale])


# In[16]:


i = 0
for col in cols_to_scale:
    df[col] = scalled[:,i]
    i += 1


# In[17]:


df.head()


# In[18]:


plt.figure(figsize=(10,10))
sns.pairplot(data=df)
plt.show()


# In[19]:


df.drop('Avatar',axis=1, inplace=True)


# In[20]:


x, y = df.drop(['Yearly Amount Spent'],axis=1),df['Yearly Amount Spent']
x.shape,y.shape


# In[21]:


models = [LinearRegression(), Ridge(), Lasso(), KNeighborsRegressor(), SVR()]


# In[22]:


for model in models:
    print("Model:",model)
    print(cross_val_score(model, x, y, cv=5))
    print('\n')


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[24]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[25]:


model.score(x_test, y_test)


# In[26]:


model.score(x_train, y_train)


# In[27]:


y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)


# In[28]:


test = pd.DataFrame({
    'Y test':y_test,
    'Y test predicted':y_pred_test
})

train = pd.DataFrame({
    'Y train':y_train,
    'Y train predicted':y_pred_train
})


# In[29]:


plt.figure(figsize=(7,7))
sns.scatterplot(data=test, x='Y test', y='Y test predicted')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Test Data")
plt.show()


# In[30]:


plt.figure(figsize=(7,7))
sns.scatterplot(data=train, x='Y train', y='Y train predicted')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Train Data")
plt.show()


# In[31]:


test.sample(10)


# In[32]:


train.sample(10)


# In[ ]:




