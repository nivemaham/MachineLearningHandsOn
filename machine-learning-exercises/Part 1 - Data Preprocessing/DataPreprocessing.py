#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as py
import matplotlib as plot
import pandas as pd


# In[3]:


dataset = pd.read_csv('Data.csv')


# In[60]:


matrixOfFeature = dataset.iloc[:, :-1].values # extract all the columns except the last


# In[13]:


dependentVariableVector = dataset.iloc[:, 3].values


# In[25]:


dependentVariableVector


# In[19]:


from sklearn.preprocessing import Imputer 


# In[20]:


imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)


# In[61]:


imputer = imputer.fit(matrixOfFeature[:, 1:3])


# In[62]:


matrixOfFeature[:, 1:3] = imputer.transform(matrixOfFeature[:, 1:3])


# In[63]:


matrixOfFeature


# In[46]:


from sklearn.preprocessing import LabelEncoder


# In[47]:


lableEncoder = LabelEncoder() # encoded the country names into a number


# In[64]:


encodedValuesOfCountry = lableEncoder.fit_transform(matrixOfFeature[:, 0])


# In[65]:


matrixOfFeature[:, 0] = encodedValuesOfCountry


# In[66]:


matrixOfFeature # now the column 0 should be dummy encoded since these numbers should not be compared naturally


# In[51]:


from sklearn.preprocessing import OneHotEncoder


# In[69]:


oneHotEncoder = OneHotEncoder(categorical_features=[0])


# In[70]:


matrixOfFeature = oneHotEncoder.fit_transform(matrixOfFeature).toarray()


# In[71]:


matrixOfFeature


# In[72]:


lableEncoderOfDepVar = LabelEncoder()


# In[77]:


dependentVariableVectorEncoded = lableEncoderOfDepVar.fit_transform(dependentVariableVector)


# In[78]:


dependentVariableVectorEncoded # now the data is ready for spliting


# In[79]:


# define vars for training and test sets, assume matrix of feature is X and dependent variable is y e.g y = an equation of x


# In[81]:


from sklearn.model_selection import train_test_split


# In[82]:


x_train, x_test, y_train, y_test = train_test_split(matrixOfFeature, dependentVariableVectorEncoded, test_size = 0.2, random_state =0)


# In[87]:


from sklearn.preprocessing import StandardScaler


# In[88]:


scaler_x = StandardScaler()


# In[89]:


x_train = scaler_x.fit_transform(x_train)


# In[90]:


x_test = scaler_x.transform(x_test)


# In[93]:


x_train


# In[95]:


get_ipython().run_cell_magic('writefile', 'test.py', '')


# In[ ]:




