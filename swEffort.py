#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
from scipy.io import arff
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Formatação mais bonita para os notebooks
import seaborn as sns
import matplotlib.pyplot as plt

#matplotlib inline
#plt.style.use('fivethirtyeight')
#plt.rcParams['figure.figsize'] = (15,5)


# In[3]:


#importing or loading the dataset
df_dataset = pd.read_csv('C:/Users/Chury Sungita/Desktop/sw-effort-predictive-analysis/Datasets/sw_dataset.csv')


# In[4]:


#The DataFrame.head() function in Pandas, by default, shows you the top 5 rows of data in the DataFram
df_dataset.head()


# In[12]:


df_dataset.info()


# In[5]:


#describe() helps to get a basic insight of the dataset with min and max 
#values along with mean, median, standard deviation & several others.

df_dataset.describe()


# In[10]:


#corr() is used to find the pairwise correlation of all columns in the dataframe. 
#Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored
df_dataset.corr()


# In[6]:


#The plot() function in pyplot module of matplotlib library is used to make a 2D hexagonal binning plot of points x, y
#The show() function in pyplot module of matplotlib library is used to display all figures.

df_dataset.plot()

plt.show()


# In[8]:


#Feature selection and extraction
#Feature selection can be done in multiple ways but there are broadly 3 categories of it:
#1. Filter Method
#2. Wrapper Method
#3. Embedded Method
#We are going to use  Filter Method:

#In this stage, we filter and take only the subset of the relevant features. The model is built after selecting the features. 
#The filtering here is done using correlation matrix and it is most commonly done using Pearson correlation.


# In[7]:


#Using Pearson Correlation
#Here we will first plot the Pearson correlation heatmap and 
#see the correlation of independent variables with the output variable ['Effort']
colormap = plt.cm.viridis
plt.figure(figsize=(10,10))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.set(font_scale=1.05)
sns.heatmap(df_dataset.drop(['id'], axis=1).astype(float).corr(),linewidths=0.1,vmax=1.0, square=True,cmap=colormap, linecolor='white', annot=True)


# In[8]:


#Split the data
#Listing and defining features in array , an attribute ['id'] has been dropped from above pearson correlation
features = [ 'TeamExp', 'ManagerExp', 'YearEnd', 'Length', 'Transactions', 'Entities',
        'PointsNonAdjust', 'Adjustment', 'PointsAjust']

#will only select features which has correlation of above 0.5 (taking absolute or approach value) with the output variable.
#The correlation coefficient has values between -1 to 1
# A value closer to 0 implies weaker correlation (exact 0 implying no correlation)
# A value closer to 1 implies stronger positive correlation
# A value closer to -1 implies stronger negative correlation

max_corr_features = ['Length', 'Transactions', 'Entities','PointsNonAdjust','PointsAjust']

##Create x and y variables.
X = df_dataset[max_corr_features]
y = df_dataset['Effort']


# In[9]:


df_dataset.shape


# In[33]:


# split the dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =30)
print('Training set shape: ', X_train.shape, y_train.shape)
print('Testing set shape: ', X_test.shape, y_test.shape)


# In[44]:


#Traing set and Test set splitting
#1.KNEIGHBORSREGRESSION
#random_state is a parameter to fix the way the data is being sampled. Therefore, if you want to reproduce the same model
#you choose any value for random_state and next time you run your code you will get the same data split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

neigh = KNeighborsRegressor(n_neighbors=3, weights='uniform')
neigh.fit(X_train, y_train) 
print(neigh.score(X_test, y_test))


# In[45]:


#1.KNEIGHBORSREGRESSION
#EVALUATION OF MODEL KNeighborsRegressor REGRESSION
#finally, if we execute this then our model will be ready, now we have x_test data we use this data for the prediction of 
#max_features

#Now, we have to compare the y_prediction values with the
#original values because we have to calculate the accuracy of our model, which was implemented by a concept called r2_score.
x_prediction =  neigh.predict(X_test)
x_prediction


# In[47]:


#1.KNEIGHBORSREGRESSION
# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,x_prediction)
print('r2 score is',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,x_prediction))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,x_prediction)))


# In[ ]:





# In[28]:


#2.LINEAR REGRESSION
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


# In[43]:


#EVALUATION OF MODEL LINEAR REGRESSION
#finally, if we execute this then our model will be ready, now we have x_test data we use this data for the prediction of 
#max_features

#Now, we have to compare the y_prediction values with the
#original values because we have to calculate the accuracy of our model, which was implemented by a concept called r2_score.
y_prediction =  model.predict(X_test)
y_prediction


# In[40]:


#2.LINEAR REGRESSION
# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_prediction)
print('r2 score is',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))


# In[ ]:





# In[39]:


plt.figure(figsize=(18,6))
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.loc']= 'upper left'
plt.rcParams['axes.labelsize']= 32

for i, feature in enumerate(max_corr_features):
   
    # Knn Regression Model 
    xs, ys = zip(*sorted(zip(X_test[feature], neigh.fit(X_train, y_train).predict(X_test))))
    
    # Linear Regression Model 
    model_xs, model_ys = zip(*sorted(zip(X_test[feature], model.fit(X_train, y_train).predict(X_test))))
    

    plt.scatter(X_test[feature], y_test, label='Real data', lw=2,alpha= 0.7, c='k' )
    plt.plot(model_xs, model_ys , lw=2, label='Linear Regression Model', c='cornflowerblue')
    plt.plot(xs, ys , lw=2,label='K Nearest Neighbors (k=3)', c='yellowgreen')
   # plt.plot(svc_model_xs, svc_model_ys , lw=2,label='Support Vector Machine (Kernel=Linear)', c='gold')
    
    plt.xlabel(feature)
    plt.ylabel('Effort')
    plt.legend()
    plt.show()


# In[ ]:




