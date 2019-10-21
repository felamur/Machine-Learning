#!/usr/bin/env python
# coding: utf-8

# In[264]:


import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings


# In[265]:


warnings.filterwarnings('ignore')


# In[266]:


data=pd.read_csv('WomboCombo2.csv')
print (len(data))


# In[267]:


no_unknown_accepted = ['Age', 'Resting Blood Pressure',
       'Serum Cholestoral', 'Max Heart Rate',
       'ST Depression Induced by Exercise']

no_unknown_accepted_but_also_cant_be_fractional = ['Sex', 'Chest Pain', 'Fasting Blood Sugar', 'Electrocardiographic',
                                                  ' Exercise Induced Angina', 'Slope of the Peak Exercise ST Segment',
                                                  'Number of Major Vessels', 'Thal', 'Num']

Alll = ['Age', 'Resting Blood Pressure',
       'Serum Cholestoral', 'Max Heart Rate',
       'ST Depression Induced by Exercise','Sex', 'Chest Pain', 'Fasting Blood Sugar', 'Electrocardiographic',
                                                  ' Exercise Induced Angina', 'Slope of the Peak Exercise ST Segment',
                                                  'Number of Major Vessels', 'Thal', 'Num']


#Replace question mark with nan and check which columns have it
for cols in Alll:
    data[cols] = data[cols].replace('?', np.NaN)
    
#Keep only rows with at least 8 non-NA values (6 nan per row) 
# 7 is half, row needs to have more than half non-na vals
data = data.dropna(thresh=8)

#Fix data so 120.0 and 120 counts as one etc...
for column in Alll:
    data[column] = round((pd.to_numeric(data[column], errors='coerce')),1)


#Make zeros nan    
data['Max Heart Rate'] = data['Max Heart Rate'].replace(0.0, np.NaN)    
data['Resting Blood Pressure'] = data['Resting Blood Pressure'].replace(0.0, np.NaN)
data['Serum Cholestoral'] = data['Serum Cholestoral'].replace(0.0, np.NaN)
     
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


print('Number of missing data',data.isnull().sum().sum())
print(data.isnull().sum())


minAge=min(data.Age)
maxAge=max(data.Age)
meanAge=data.Age.mean()

#Make 3,6,7 to 0,1,2
data['Thal'] = data['Thal'].replace(3.0, 0)
data['Thal'] = data['Thal'].replace(6.0, 1)
data['Thal'] = data['Thal'].replace(7.0, 2) 
    
 

sns.countplot("Num", data=data)

#Gather 2,3,4 under 1, same
column = 'Num'
data[column] = data[column].replace(2, 1)
data[column] = data[column].replace(3, 1)
data[column] = data[column].replace(4, 1)


  
#This one fills empty data according to age and sex of patient (mean)
youngFemale = data[(data.Age>=28)&(data.Age<40)&(data.Sex==0.0)]
middleFemale = data[(data.Age>=40)&(data.Age<53)&(data.Sex==0.0)]
oldFemale = data[(data.Age>53)&(data.Sex==0.0)]

youngMale = data[(data.Age>=28)&(data.Age<40)&(data.Sex==1.0)]
middleMale = data[(data.Age>=40)&(data.Age<53)&(data.Sex==1.0)]
oldMale = data[(data.Age>53)&(data.Sex==1.0)] 

for columnG in no_unknown_accepted:
    mean = round((pd.to_numeric(youngFemale[columnG], errors='coerce')).mean(skipna=True),1)
    youngFemale[columnG] = youngFemale[columnG].replace(np.NaN, mean)
for columnG in no_unknown_accepted:
    mean = round((pd.to_numeric(middleFemale[columnG], errors='coerce')).mean(skipna=True),1)
    middleFemale[columnG] = middleFemale[columnG].replace(np.NaN, mean)
for columnG in no_unknown_accepted:
    mean = round((pd.to_numeric(oldFemale[columnG], errors='coerce')).mean(skipna=True),1)
    oldFemale[columnG] = oldFemale[columnG].replace(np.NaN, mean)
for columnG in no_unknown_accepted:
    mean = round((pd.to_numeric(youngMale[columnG], errors='coerce')).mean(skipna=True),1)
    youngMale[columnG] = youngMale[columnG].replace(np.NaN, mean)
for columnG in no_unknown_accepted:
    mean = round((pd.to_numeric(middleMale[columnG], errors='coerce')).mean(skipna=True),1)
    middleMale[columnG] = middleMale[columnG].replace(np.NaN, mean)
for columnG in no_unknown_accepted:
    mean = round((pd.to_numeric(oldMale[columnG], errors='coerce')).mean(skipna=True),1)
    oldMale[columnG] = oldMale[columnG].replace(np.NaN, mean)
    
for columnG in no_unknown_accepted_but_also_cant_be_fractional:
    mean = round((pd.to_numeric(youngFemale[columnG], errors='coerce')).mean(skipna=True),1)
    h = int(round(mean))
    youngFemale[columnG] = youngFemale[columnG].replace(np.NaN, h)
for columnG in no_unknown_accepted_but_also_cant_be_fractional:
    mean = round((pd.to_numeric(middleFemale[columnG], errors='coerce')).mean(skipna=True),1)
    h = int(round(mean))
    middleFemale[columnG] = middleFemale[columnG].replace(np.NaN, h)
for columnG in no_unknown_accepted_but_also_cant_be_fractional:
    mean = round((pd.to_numeric(oldFemale[columnG], errors='coerce')).mean(skipna=True),1)
    h = int(round(mean))
    oldFemale[columnG] = oldFemale[columnG].replace(np.NaN, h)
for columnG in no_unknown_accepted_but_also_cant_be_fractional:
    mean = round((pd.to_numeric(youngMale[columnG], errors='coerce')).mean(skipna=True),1)
    h = int(round(mean))
    youngMale[columnG] = youngMale[columnG].replace(np.NaN, h)
for columnG in no_unknown_accepted_but_also_cant_be_fractional:
    mean = round((pd.to_numeric(middleMale[columnG], errors='coerce')).mean(skipna=True),1)
    h = int(round(mean))
    middleMale[columnG] = middleMale[columnG].replace(np.NaN, h)
for columnG in no_unknown_accepted_but_also_cant_be_fractional:
    mean = round((pd.to_numeric(oldMale[columnG], errors='coerce')).mean(skipna=True),1)
    h = int(round(mean))
    oldMale[columnG] = oldMale[columnG].replace(np.NaN, h)    

   
frames = [youngFemale, middleFemale, oldFemale, youngMale, middleMale, oldMale]

data = pd.concat(frames)

print (len(data))
print('Number of missing data',data.isnull().sum().sum())  


  


# In[268]:


sns.countplot("Num", data=data)
print(data.Num.value_counts())


# In[269]:


#SPLIT THE SET FIRST

dataX=data.drop('Num',axis=1)
dataY=data['Num']

X_train,X_test,y_train,y_test=train_test_split(dataX,dataY,test_size=0.2,random_state=0)

print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)


# In[270]:


# applying SMOTE to our data and checking the class counts
print('Original dataset shape %s' % Counter(y_test))


sm = SMOTE(random_state=0)
X_res, y_res = sm.fit_resample(X_test, y_test)
print(sorted(Counter(y_res).items()))
data_res = np.column_stack((X_res, y_res))


# In[271]:


#print('Original dataset shape %s' % Counter(y_test))

# random oversampling
#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_resample(X_test, y_test)

# using Counter to display results of naive oversampling
#print(sorted(Counter(y_resampled).items()))
#print('Resampled dataset shape %s' % Counter(y_resampled))


# In[272]:


#print('Original dataset shape %s' % Counter(y_test))


#rus = RandomUnderSampler(random_state=0)
#X_res, y_res = rus.fit_resample(X_test, y_test)
#print('Resampled dataset shape %s' % Counter(y_res))


# In[273]:


y_test = y_res
X_test = X_res

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[274]:


pca = PCA(.95)
pca.fit(X_train) 
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# In[275]:


parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


gslog=GridSearchCV(SVC(),parameters,scoring='accuracy')
gslog.fit(X_train,y_train)
print('Best parameters set:')
print(gslog.best_params_)
classifier = gslog

y_pred = gslog.predict(X_test)
# Predicting the Test set results
#y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print (cm)

cm_test = confusion_matrix(y_pred, y_test)

f1score= f1_score(y_test, y_pred)
accuracy= accuracy_score(y_test, y_pred)

y_pred_train = classifier.predict(X_train)
accuracy2 = accuracy_score(y_pred_train, y_train)


print('Accuracy for traning set for svm = ')
print(accuracy2)
print('Accuracy for testing set for svm = ')
print(accuracy)
print('F1 for test set for svm = ')
print(f1score)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()

print(classification_report(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))


# In[276]:


parameters=[
{
    'penalty':['l1','l2'],
    'C':[0.1,0.4,0.5],
    'random_state':[0]
    },
]

gslog=GridSearchCV(LogisticRegression(),parameters,scoring='accuracy')
gslog.fit(X_train,y_train)
print('Best parameters set:')
print(gslog.best_params_)
classifier = gslog


y_pred = gslog.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print (cm)

cm_test = confusion_matrix(y_pred, y_test)

f1score= f1_score(y_test, y_pred)
accuracy= accuracy_score(y_test, y_pred)

y_pred_train = classifier.predict(X_train)
accuracy2 = accuracy_score(y_pred_train, y_train)

crossVal=cross_val_score(estimator=LogisticRegression(),X=X_train,y=y_train,cv=12)
print(crossVal.mean())
print(crossVal.std())


print('Accuracy for traning set for Logistic Regression = ')
print(accuracy2)
print('Accuracy for testing set for Logistic Regression = ')
print(accuracy)
print('F1 for test set for Logistic Regression = ')
print(f1score)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()

plt.figure(2)

y_proba=gslog.predict_proba(X_test)

falseP_rate, trueP_rate, thresholds = roc_curve(y_test,y_proba[:,1])
roc_auc = auc(falseP_rate, trueP_rate)
plt.plot(falseP_rate, trueP_rate , 'b', label = 'AUC' %roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
print(roc_auc)
print(classification_report(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))


# In[277]:


math.sqrt(len(y_train))
parameters=[
{
    'n_neighbors':np.arange(2,33),
    'n_jobs':[2,6]
    },
]

gslog=GridSearchCV(KNeighborsClassifier(),parameters,scoring='accuracy')
gslog.fit(X_train,y_train)
print('Best parameters set:')
print(gslog.best_params_)
classifier = gslog

y_pred = gslog.predict(X_test)




cm = confusion_matrix(y_pred, y_test)
print (cm)

f1score= f1_score(y_test, y_pred)
accuracy= accuracy_score(y_test, y_pred)

y_pred_train = classifier.predict(X_train)
accuracy2 = accuracy_score(y_pred_train, y_train)


print('Accuracy for traning set for KNN = ')
print(accuracy2)
print('Accuracy for testing set for KNN = ')
print(accuracy)
print('F1 for test set for KNN = ')
print(f1score)
plt.title("KNN Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()

plt.figure(2)

y_proba=gslog.predict_proba(X_test)

falseP_rate, trueP_rate, thresholds = roc_curve(y_test,y_proba[:,1])
roc_auc = auc(falseP_rate, trueP_rate)
plt.plot(falseP_rate, trueP_rate , 'b', label = 'AUC' %roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
print(roc_auc)
print(classification_report(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))


# In[278]:


classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print (cm)

cm_test = confusion_matrix(y_pred, y_test)

f1score= f1_score(y_test, y_pred)
accuracy= accuracy_score(y_test, y_pred)

y_pred_train = classifier.predict(X_train)
accuracy2 = accuracy_score(y_pred_train, y_train)


print('Accuracy for traning set for Naive = ')
print(accuracy2)
print('Accuracy for testing set for Naive = ')
print(accuracy)
print('F1 for test set for Naive = ')
print(f1score)
plt.title("Naive Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()

plt.figure(2)
y_proba=gslog.predict_proba(X_test)

falseP_rate, trueP_rate, thresholds = roc_curve(y_test,y_proba[:,1])
roc_auc = auc(falseP_rate, trueP_rate)
plt.plot(falseP_rate, trueP_rate , 'b', label = 'AUC' %roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
print(roc_auc)
print(classification_report(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))


# In[279]:


parameters = {'max_depth': np.arange(3, 10)}

gslog=GridSearchCV(DecisionTreeClassifier(),parameters,scoring='accuracy')
gslog.fit(X_train,y_train)
print('Best parameters set:')
print(gslog.best_params_)
classifier = gslog

y_pred = gslog.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print (cm)

cm_test = confusion_matrix(y_pred, y_test)

f1score= f1_score(y_test, y_pred)
accuracy= accuracy_score(y_test, y_pred)

y_pred_train = classifier.predict(X_train)
accuracy2 = accuracy_score(y_pred_train, y_train)


print('Accuracy for traning set for Decision Tree = ')
print(accuracy2)
print('Accuracy for testing set for Decision Tree = ')
print(accuracy)
print('F1 for test set for Decision Tree = ')
print(f1score)
plt.title("Decision Tree Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()

plt.figure(2)
y_proba=gslog.predict_proba(X_test)

falseP_rate, trueP_rate, thresholds = roc_curve(y_test,y_proba[:,1])
roc_auc = auc(falseP_rate, trueP_rate)
plt.plot(falseP_rate, trueP_rate , 'b', label = 'AUC' %roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
print(roc_auc)
print(classification_report(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))


# In[280]:


parameters = {'max_features': ['auto', 'sqrt', 'log2'],
 'n_estimators': [10, 50, 100, 200, 400, 700] }


gslog=GridSearchCV(RandomForestClassifier(),parameters,scoring='accuracy')
gslog.fit(X_train,y_train)
print('Best parameters set:')
print(gslog.best_params_)

y_pred = gslog.predict(X_test)


#classifier = RandomForestClassifier(n_estimators = 10)
#classifier.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print (cm)

cm_test = confusion_matrix(y_pred, y_test)

f1score= f1_score(y_test, y_pred)
accuracy= accuracy_score(y_test, y_pred)

y_pred_train = classifier.predict(X_train)
accuracy2 = accuracy_score(y_pred_train, y_train)


print('Accuracy for traning set for Random Forest = ')
print(accuracy2)
print('Accuracy for testing set for Random Forest = ')
print(accuracy)
print('F1 for test set for Random Forest = ')
print(f1score)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm,annot=True)
plt.show()

plt.figure(2)
#y_proba=classifier.predict_proba(X_test)
y_proba=gslog.predict_proba(X_test)

falseP_rate, trueP_rate, thresholds = roc_curve(y_test,y_proba[:,1])
roc_auc = auc(falseP_rate, trueP_rate)
plt.plot(falseP_rate, trueP_rate , 'b', label = 'AUC' %roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
print(roc_auc)
print(classification_report(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




