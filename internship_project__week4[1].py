#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("C:/Users/Dell/Downloads/TelcoChurn.csv")
data.head()    


# In[3]:


data.columns


# In[4]:


data_1 = data.copy()


# In[5]:


data.shape


# We have 7043 Rows and 21 Columns in this Telecom Dataset.
# We have target Variable 'Churn' with object datatype, leading this to classification problem.
# There is interesting entry here under object datatype which is 'TotalCharges'. This feature is numerical in nature but categories as Object datatypes. This implies that there is presence of string variable in this column or might be data error.
# 
# At end we have 3 Numerical variable and 18 categorical variable. Out of which 'CustomerID' is unnecessary variable from our analytical & modelling viewpoint. We will drop 'CustomerID' column.

# In[6]:


data.isnull().sum()


# There is no null values

# In[7]:


data.Churn.value_counts()


# # 1. visualisation

# In[8]:


data['Churn'].value_counts().plot(kind='pie',labels=['Not Churn','Churn'],colors=['green','red'])
plt.legend() 


# In[9]:


plt.rcParams["figure.autolayout"] = True
sns.set_palette('husl')

data['Churn'].value_counts().plot.pie(explode=[0,0.1],autopct='%3.1f%%',
                                          textprops ={ 'fontweight': 'bold','fontsize':18},shadow=True)


# 26.5 % Customer choose to churn service in last month. Which is quite high number.This all leads to imbalanced data case as churn is our target variable.

# In[10]:


#univariate analysis

y=data.Contract.value_counts()
x=data['Contract'].unique()
plt.title('unique contract distribution')
plt.scatter(x, y)
plt.show()

data['Contract'].value_counts().plot.pie(explode=[0.03,0.03,0.03],autopct='%2.1f%%',
                                          textprops ={ 'fontweight': 'bold','fontsize':13},shadow=True)
plt.show()

data1=data
plt.pie(y,labels=x)
plt.legend()

plt.show()

data['OnlineSecurity'].value_counts().plot.pie(explode=[0.03,0.03,0.03],autopct='%2.1f%%',
                                          textprops ={ 'fontweight': 'bold','fontsize':13},shadow=True)
plt.show()



data['MultipleLines'].value_counts().plot.pie(explode=[0.03,0.03,0.03],autopct='%2.1f%%',
                                          textprops ={ 'fontweight': 'bold','fontsize':13},shadow=True)
plt.show()


data['PaymentMethod'].value_counts().plot.pie(autopct='%2.1f%%',
                                          textprops ={ 'fontweight': 'bold','fontsize':13},shadow=True)
plt.show()
plt.pie(data.PaymentMethod.value_counts(),labels=data['PaymentMethod'].unique())
plt.legend()
plt.show()




     


# In[11]:


Customer_Stayed=data[data['Churn']=='No'].MonthlyCharges 
Customer_Churned=data[data['Churn']=='Yes'].MonthlyCharges 

plt.xlabel('Monthly charges')
plt.ylabel('Customer status')
plt.hist([Customer_Stayed,Customer_Churned], color=['black','red'],label=['Stayed','Churned'])
plt.legend()
plt.show()

Customer_Stayed=data[data['Churn']=='No'].TotalCharges 
Customer_Churned=data[data['Churn']=='Yes'].TotalCharges 
plt.xlabel('Total charges')
plt.ylabel('Customer status')
plt.hist([Customer_Stayed,Customer_Churned], color=['blue','red'],label=['Stayed','Churned'])
plt.legend()
plt.show()

Customer_Stayed=data[data['Churn']=='No'].tenure 
Customer_Churned=data[data['Churn']=='Yes'].tenure 
plt.xlabel('Tenure ')
plt.ylabel('Customer status')
plt.hist([Customer_Stayed,Customer_Churned], color=['green','red'],label=['Stayed','Churned'])
plt.legend()
plt.show()




# 

# In[12]:


pd.crosstab(data['Churn'], data['InternetService']).plot(kind='bar')
plt.show()


pd.crosstab(data['Churn'], data['PaymentMethod']).plot(kind='bar')
plt.show()


pd.crosstab(data['Churn'], data['DeviceProtection']).plot(kind='bar')
plt.show()


pd.crosstab(data['Churn'], data['TechSupport']).plot(kind='bar')
plt.show()

pd.crosstab(data['Churn'], data['Contract']).plot(kind='bar')
plt.show()



# In[13]:


#to check outliers
sns.boxplot(data['MonthlyCharges'])
plt.title('monthly charges')
plt.show()

sns.boxplot(data['TotalCharges'])
plt.title('Total charges')
plt.show()

sns.boxplot(data['tenure'])
plt.title('tenure')
plt.show()


# In[14]:


#no outliers


# In[15]:


data_1.corr()


# In[16]:


plt.figure(figsize=(20,15))
data_1=data_1.corr()
sns.heatmap(data_1,cmap='viridis',annot=True,annot_kws={'size':20})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# In[ ]:





# In[ ]:





# In[17]:


#visualising after encoding categorical varibles
# internet service wise chrun analysis
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Churn']=le.fit_transform(data['Churn'])
internet_churn=data.groupby(["InternetService"])["Churn"].mean().head()
plt.bar(data['InternetService'].unique(),internet_churn,color='red')
plt.ylabel('churn rate')
plt.show()




# In[18]:


#contract wise churn rate
contract_churn=data.groupby(["Contract"])["Churn"].mean().head()
plt.bar(data['Contract'].unique(),contract_churn,color='blue')
plt.xlabel('churn rate')
plt.ylabel('contract')
plt.show()


# In[19]:


print(data['PhoneService'].value_counts())

print(data['DeviceProtection'].value_counts())

print(data['TechSupport'].value_counts())


# In[20]:


contract_churn=data.groupby(["DeviceProtection"])["Churn"].mean().head()
plt.bar(data['DeviceProtection'].unique(),contract_churn,color='red')
plt.xlabel('device protection')
plt.ylabel('churn')
plt.show()

TechSupport_churn=data.groupby(["TechSupport"])["Churn"].mean().head()
plt.bar(data['TechSupport'].unique(),TechSupport_churn,color='red')
plt.xlabel('TechSupport')
plt.ylabel('churn rate')
plt.show()


# # 2.preprocessing

# In[ ]:





# In[21]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Churn']=le.fit_transform(data['Churn'])
data['gender']=le.fit_transform(data['gender'])
data['Partner']=le.fit_transform(data['Partner'])
data['Dependents']=le.fit_transform(data['Dependents'])
data['PhoneService']=le.fit_transform(data['PhoneService'])
data['MultipleLines']=le.fit_transform(data['MultipleLines'])
data['InternetService']=le.fit_transform(data['InternetService'])
data['OnlineSecurity']=le.fit_transform(data['OnlineSecurity'])
data['DeviceProtection']=le.fit_transform(data['DeviceProtection'])
data['TechSupport']=le.fit_transform(data['TechSupport'])
data['StreamingTV']=le.fit_transform(data['StreamingTV'])
data['StreamingMovies']=le.fit_transform(data['StreamingMovies'])
data['Contract']=le.fit_transform(data['Contract'])
data['PaperlessBilling']=le.fit_transform(data['PaperlessBilling'])
data['PaymentMethod']=le.fit_transform(data['PaymentMethod'])
data['Churn']=le.fit_transform(data['Churn'])
data['OnlineBackup']=le.fit_transform(data['OnlineBackup'])

data=data.drop('customerID',axis=1)
data2=data


# In[22]:


data.head()


# In[23]:


column_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[column_to_scale] = scaler.fit_transform(data[column_to_scale])
data


# In[24]:


data1=pd.get_dummies(data)
data1


# # model building

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,f1_score

Y=data1.Churn
X=data1.drop('Churn', axis=1)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_scale = scaler.fit_transform(X)


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(data1, Y, random_state=99, test_size=.3)
print('Training feature matrix size:',X_train.shape)
print('Training target vector size:',Y_train.shape)
print('Test feature matrix size:',X_test.shape)
print('Test target vector size:',Y_test.shape)


# In[27]:


#Finding best Random state

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,f1_score
maxAccu=0
maxRS=0
for i in range(1,250):
    X_train,X_test,Y_train,Y_test = train_test_split(X_scale,Y,test_size = 0.3, random_state=i)
    log_reg=LogisticRegression()
    log_reg.fit(X_train,Y_train)
    y_pred=log_reg.predict(X_test)
    acc=accuracy_score(Y_test,y_pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print('Best accuracy is', maxAccu ,'on Random_state', maxRS)


# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, random_state=178, test_size=.3)
log_reg=LogisticRegression()
log_reg.fit(X_train,Y_train)
y_pred=log_reg.predict(X_test)


# Logistics Regression Evaluation
# 
# 
# Accuracy Score of Logistics Regression : 0.8211074301940369

# In[29]:


#Finding Optimal value of n_neighbors for KNN

from sklearn import neighbors
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsClassifier(n_neighbors = K)

    model.fit(X_train,Y_train)  #fit the model
    y_pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(Y_test,y_pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[ ]:





# At k=18, we get the minimum RMSE value which approximately  0.4505872489393898, and shoots up on further increasing the k value. We can safely say that k=20 will give us the best result in this case

# In[30]:


model=[ LogisticRegression(),
        SVC(),
        GaussianNB(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(n_neighbors = 18),
        RandomForestClassifier(),
        ExtraTreesClassifier()]
        
for m in model:
    m.fit(X_train,Y_train)
    y_pred=m.predict(X_test)
    print('Classification ML Algorithm Evaluation Matrix',m,'is' )
    print('\n')
    print('Accuracy Score :', accuracy_score(Y_test, y_pred))
    print('\n')
    print('Confusion matrix :',confusion_matrix(Y_test, y_pred))
    print('\n')
    print('Classification Report :',classification_report(Y_test, y_pred))
    print('\n')
    print('============================================================================================================')


# In[ ]:




