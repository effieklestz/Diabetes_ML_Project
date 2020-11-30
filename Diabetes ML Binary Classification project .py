#!/usr/bin/env python
# coding: utf-8

# In[221]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[222]:


# Ingesting The data from CSV 
diabetes_data = pd.read_csv('diabetes.csv')
diabetes_data.head()


# In[223]:


# Basic Statistics <== 0 values need to be replaced with NaN Values
diabetes_data.describe().T


# In[224]:


#Replacing 0's in Glucose, BloodPressure, SkinThickness, BMI, Insulin columns
diabetes_data[['Glucose','BloodPressure','SkinThickness','BMI','Insulin',]] = diabetes_data[['Glucose','BloodPressure','SkinThickness','BMI','Insulin',]].replace(0.0, np.NaN)


# In[225]:


diabetes_data.describe().T


# In[227]:


# inspecting the % of in each column
def pecent_NaN():
    NUll_data = pd.DataFrame(diabetes_data.isnull().sum(), columns=['NaNs'])
NUll_data['Count'] = 768
NUll_data['Percent_Nans'] = NUll_data['null']/ NUll_data['count']*100
columns= diabetes_data.columns

ax = sns.barplot(x ='Percent_Nans' , y= columns , data= NUll_data)
plt.title('% Nan')


# In[231]:


# Plan = replace Nans with the mean values of their respective Outcome
mean_data = pd.DataFrame(diabetes_data.groupby('Outcome').mean(), columns= columns.drop('Outcome'))
mean_data.head()


# In[ ]:


#Replaceing Nans


# In[139]:


diabetes_data.loc[(diabetes_data['Outcome'] == 0 ) & (diabetes_data['Glucose'].isnull()), 'Glucose'] = 110
diabetes_data.loc[(diabetes_data['Outcome'] == 1 ) & (diabetes_data['Glucose'].isnull()), 'Glucose'] = 142
diabetes_data.isnull().sum()


# In[141]:


diabetes_data.loc[(diabetes_data['Outcome'] == 0 ) & (diabetes_data['BloodPressure'].isnull()), 'BloodPressure'] = 70
diabetes_data.loc[(diabetes_data['Outcome'] == 1 ) & (diabetes_data['BloodPressure'].isnull()), 'BloodPressure'] = 75
diabetes_data.isnull().sum()


# In[143]:


diabetes_data.loc[(diabetes_data['Outcome'] == 0 ) & (diabetes_data['SkinThickness'].isnull()), 'SkinThickness'] = 27
diabetes_data.loc[(diabetes_data['Outcome'] == 1 ) & (diabetes_data['SkinThickness'].isnull()), 'SkinThickness'] = 33
diabetes_data.isnull().sum()


# In[145]:


diabetes_data.loc[(diabetes_data['Outcome'] == 0 ) & (diabetes_data['Insulin'].isnull()), 'Insulin'] = 130
diabetes_data.loc[(diabetes_data['Outcome'] == 1 ) & (diabetes_data['Insulin'].isnull()), 'Insulin'] = 206
diabetes_data.isnull().sum()


# In[146]:


diabetes_data.loc[(diabetes_data['Outcome'] == 0 ) & (diabetes_data['BMI'].isnull()), 'BMI'] = 30
diabetes_data.loc[(diabetes_data['Outcome'] == 1 ) & (diabetes_data['BMI'].isnull()), 'BMI'] = 35
diabetes_data.isnull().sum()


# In[148]:


columns = list(diabetes_data.columns)
columns.remove('Outcome')

for i in columns:
    sns.catplot(x='Outcome', y= i , data=diabetes_data, kind='box')
    


# In[ ]:





# In[ ]:


sns.catplot(x="size", y="total_bill", data=tips)


# In[81]:


import matplotlib.pyplot as plt
p = diabetes_data.hist(figsize=(20,20))



# In[156]:


diabetes_data.dtypes


# In[157]:


print(diabetes_data['Outcome'].value_counts())
plt.figure(figsize=(15,12)) 
sns.heatmap(diabetes_data.corr(), square=True,annot= True, cmap='RdYlGn')


# In[158]:


sns.pairplot(diabetes_data , hue='Outcome')
    
    


# In[159]:


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

y = diabetes_data.Outcome
X = diabetes_data.drop('Outcome', axis= 1)


# In[160]:


X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state= 40)

sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[161]:


from sklearn.neighbors import KNeighborsClassifier

train_scores= []
test_scores = []

for i in range(1,12):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))

max_train_score= max(train_scores)  
max_test_score = max(test_scores)
print('Max test score {}'.format(max_test_score))


# In[ ]:





# In[162]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,12),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,12),test_scores,marker='o',label='Test Score')


# In[163]:


from sklearn.metrics import confusion_matrix

y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[164]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[235]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=0.01,random_state=2)
rf.fit(X_train,y_train)

print("Score:" + str(rf.score(X_test,y_test)))


features = list(X.columns)
importances = pd.Series(data=rf.feature_importances_,index= features)
importances_sorted = importances.sort_values()
top_importance = importances_sorted[importances_sorted > 0]

# barplot of importances
top_importance.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.yticks(horizontalalignment='right',fontweight='light')


plt.show()


# In[ ]:




