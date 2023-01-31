#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import sklearn.metrics as sm


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


diabetes_dataset = pd.read_csv('diabetes_clean.csv')


# In[9]:


diabetes_dataset


# In[10]:


diabetes_dataset.describe()


# In[13]:


sum(diabetes_dataset.isnull().sum())


# In[14]:


print((diabetes_dataset[['glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age']]== 0).sum())


# In[15]:


diabetes_dataset[['glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age']] = diabetes_dataset[['glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'dpf', 'age']].replace(0, np.NAN)


# In[16]:


print(diabetes_dataset.isnull().sum())


# In[17]:


diabetes_dataset.fillna(diabetes_dataset.mean(), inplace=True)


# In[18]:


print(diabetes_dataset.isnull().sum())


# In[19]:


diabetes_dataset


# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


labelencoder = LabelEncoder()


# In[23]:


dataTransform = diabetes_dataset.copy()


# In[27]:


for data in diabetes_dataset.columns:
    dataTransform[data] = labelencoder.fit_transform(diabetes_dataset[data])


# In[28]:


dataTransform


# In[29]:


x = dataTransform.drop(['diabetes'], axis=1)


# In[30]:


x


# In[31]:


y = dataTransform['diabetes']


# In[32]:


y


# In[34]:


diabetes_feature_list = list(x.columns)


# In[35]:


diabetes_feature_list


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 41)


# In[41]:


y_train


# In[42]:


x_train.shape


# In[43]:


y_train.shape


# In[44]:


x_test.shape


# In[45]:


y_test.shape


# In[47]:


from sklearn.ensemble import RandomForestClassifier


# In[53]:


randomforestclassifier = RandomForestClassifier(n_estimators=450)


# In[54]:


randomforestclassifier.fit(x_train,y_train)


# In[56]:


prediction_y = randomforestclassifier.predict(x_test)


# In[57]:


prediction_y


# In[59]:


experiment_accuracy = sm.accuracy_score(y_test, prediction_y)
print('Accuracy score is:' , str(experiment_accuracy))


# In[60]:


from sklearn import metrics


# In[61]:


print("Classification Report : " , metrics.classification_report(prediction_y,y_test,target_names=["Diabetes","No Diabetes"]))


# In[62]:


from sklearn.metrics import confusion_matrix


# In[63]:


import seaborn as sb


# In[64]:


sb.set()


# In[66]:


get_ipython().run_line_magic('matplotlib','inline')


# In[67]:


import matplotlib.pyplot as pt


# In[68]:


confusionmt = confusion_matrix(y_test,prediction_y)


# In[71]:


sb.heatmap(confusionmt.T, square=True, annot=True,fmt='d', cbar=False)


# In[74]:


pt.xlabel('true class axis')
pt.ylabel('predicted class axis')


# In[ ]:




