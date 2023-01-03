#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary liberraries and import dataset to perform data analytics work


# In[9]:


import os
os.getcwd()


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

"""
import warnings
warnings.filterwarnings('ignore')
"""


# In[18]:


data = pd.read_csv(r'C:\Users\jaswanth\Downloads\learnbay\projects\project_24-march-by sundaram\manufacturing dataset.csv')
data.head()


# In[ ]:





# ################################################

# In[19]:


pd.set_option("display.max_rows", 600)
pd.set_option('display.max_columns', 1000)
pd.set_option("display.width", 1000)


# In[20]:


data = pd.read_csv("manufacturing dataset.csv")
dataset = pd.DataFrame(data)
dataset.head()


# In[21]:


# Pre-processing 
## part 1 : handling missing value
## part 2 : handling Encoding(label encoder, one-hot-encoder and dummy variable)
## part 3 : handling outlier
## part 4 : Feature scaling (standarisation, normalisation, min_max)
## part 5 : handling imbalance dataset - only for classification problem


# In[22]:


dataset.shape


# In[23]:


dataset.columns


# In[24]:


dataset.columns = 'features_'+dataset.columns


# In[25]:


dataset.head(1)


# In[26]:


dataset.rename(columns = {'features_Time': 'Time'}, inplace=True)
dataset.rename(columns = {'features_Pass/Fail': 'Pass_Fail'}, inplace=True)


# In[27]:


dataset.head(1)


# In[28]:


dataset.info()


# In[29]:


dataset.dtypes


# In[30]:


dataset.describe().transpose()


# In[31]:


dataset.isnull().sum()


# In[32]:


dataset.isna().sum()


# In[ ]:





# In[33]:


df = dataset.iloc[:,1:]
df = df.apply(lambda x:x.fillna(0), axis=0)
# axis=0 - row wise
df2 = dataset.iloc[:,0]
result = pd.concat([df, df2], axis=1).reindex(df.index)
# axis=1


# In[34]:


result.head(2)


# In[35]:


result.isnull().any()


# In[36]:


result['Pass_Fail'].value_counts(normalize=True)


# In[37]:


result['Pass_Fail'].value_counts()


# In[38]:


result['Pass_Fail'].value_counts().plot(kind='bar')


# In[39]:


corr = result.corr()
print(corr)


# In[40]:


#sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[41]:


corr.to_csv("correlation.csv")


# In[42]:


result.columns


# In[43]:


from datetime import datetime
result['year'] = pd.DatetimeIndex(result['Time']).year
result['month'] = pd.DatetimeIndex(result['Time']).month
result['date'] = pd.DatetimeIndex(result['Time']).day
result['week_day'] = pd.DatetimeIndex(result['Time']).weekday
result['start_time'] = pd.DatetimeIndex(result['Time']).time
result['hour'] = pd.DatetimeIndex(result['Time']).hour
result['min'] = pd.DatetimeIndex(result['Time']).minute


# In[44]:


result.head()


# In[45]:


result.year.unique()


# In[46]:


result.month.unique()


# In[47]:


result.date.unique()


# In[48]:


result.week_day.unique()


# In[49]:


sns.displot(result[result.Pass_Fail== -1]['month'], color ='g');
sns.displot(result[result.Pass_Fail== 1]['month'], color ='r');


# In[50]:


sns.displot(result[result.Pass_Fail== -1]['date'], color ='g');
sns.displot(result[result.Pass_Fail== 1]['date'], color ='r');


# In[51]:


sns.displot(result[result.Pass_Fail== -1]['hour'], color ='g');
sns.displot(result[result.Pass_Fail== 1]['hour'], color ='r');


# In[52]:


# Pass_Fail	Time	start_time
x = result.drop(['Pass_Fail','Time','start_time','year'], axis=1)
y = result[['Pass_Fail']]


# In[53]:


x.head()


# In[54]:


y.head()


# In[55]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)


# In[56]:


x = pd.DataFrame(x_scaler, columns=x.columns[:])


# In[57]:


x.head()


# In[58]:


y.value_counts()


# In[59]:


pip install  imblearn


# In[60]:


import imblearn


# In[61]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
x_ros, y_ros = ros.fit_resample(x, y)
#from collections import Counter
#print(Counter(y))
#print("##############"*5)
#print(Counter(y_ros))


# In[62]:


y_ros.value_counts()


# In[63]:


x_ros.shape


# In[64]:


# principal component analysis - dimension reduction method
from sklearn.decomposition import PCA
pca = PCA(.95)
x_pca = pca.fit_transform(x_ros)
x_pca.shape


# In[65]:


x_pca


# # split the data into training and test

# In[94]:


from sklearn.model_selection import train_test_split,GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x_pca, y_ros, test_size=0.3, random_state=1)


# # Building Model

# In[98]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
from sklearn.metrics import f1_score, precision_score, recall_score 
from sklearn.metrics import roc_curve, auc
import time


# In[99]:


rforest = RandomForestClassifier()


# In[69]:


y_pred_train = rforest.predict(x_train)
y_pred_test = rforest.predict(x_test)


# In[70]:


print(confusion_matrix(y_train, y_pred_train))
print("##########"*10)
print(confusion_matrix(y_test, y_pred_test))


# In[71]:


print(classification_report(y_train, y_pred_train))
print("##########"*10)
print(classification_report(y_test, y_pred_test))


# In[72]:


print("Training Accuracy :", accuracy_score(y_train, y_pred_train))
print("##########"*10)
print("Test Accuracy :", accuracy_score(y_test, y_pred_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # decision tree classifier

# In[101]:


decision_tree_model=DecisionTreeClassifier( criterion='gini',max_depth=14,min_samples_split=8, min_samples_leaf=8)
decision_tree_model.fit(x_train, y_train)
y_pred_dt = decision_tree_model.predict(x_test)

decision_tree_model_score_train = decision_tree_model.score(x_train, y_train)
print("Training score: ",decision_tree_model_score_train)
decision_tree_model_score_test=decision_tree_model.score(x_test, y_test)
print("Testing score: ",decision_tree_model_score_test)


# In[102]:


df_p_ac=pd.DataFrame()
df_p_ac['actaul']=y_test
df_p_ac['predicted']=y_pred_dt
df_p_ac.head(10)


# # metrics ######

# In[103]:


print (confusion_matrix(y_test, y_pred_dt))


# In[104]:


accuracy_score(y_test, y_pred_dt)


# In[105]:


y_test_values=y_test['Pass_Fail'].values


# In[106]:


fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)


# In[107]:


plt.figure(1)
lw = 2
plt.plot(fpr_dt, tpr_dt, color='green',
         lw=lw, label='Decision Tree(AUC = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve')
plt.legend(loc="lower right")
plt.show()


# In[108]:


def create_conf_mat(y_test_values, y_pred_dt):
    """Function returns confusion matrix comparing two arrays"""
    if (len(y_test_values.shape) != len(y_pred_dt.shape) == 1):
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif (y_test_values.shape != y_pred_dt.shape):
        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index = y_test_values,
                                        columns = y_pred_dt)
        # Changed for Future deprecation of as_matrix
        test_crosstb = test_crosstb_comp.values
        return test_crosstb


# In[109]:


conf_mat = create_conf_mat(y_test_values, y_pred_dt)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs. Predicted Confusion Matrix')
plt.show()


# In[ ]:




