#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


heart_data = pd.read_csv("C:/Users/sagan/Downloads/DATASETS/heart.csv")
heart_data


# In[5]:


X = heart_data.drop('target', axis=1) 
y = heart_data['target'] 


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)


# In[8]:


y_pred = clf.predict(X_test)


# In[9]:


print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[10]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:




