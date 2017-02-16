
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import csv #to save the file
train = pd.DataFrame.from_csv('~/Documents/GIT_HUB/MNIST/train.csv',index_col=None)
#none because the index do not exist and we need to create one index in the data frame
test=pd.DataFrame.from_csv('~/Documents/GIT_HUB/MNIST/test.csv',index_col=None)
array = train.values
X = array[:,1:]
Y=array[:,0]
#this is done for test purpose
print(train.groupby('label').size())
#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[2]:

train.head()


# In[3]:

print(train.describe()) #this is to describe the data set


# In[4]:

train.tail()


# In[ ]:

'''print('X shape')
print(X.shape)
print('Y shape')
print(Y.shape)
print(set(Y))
print('X Train')
print(X_train.shape)
print('Y Train')
print(Y_train.shape)
print('X Validation')
print(X_validation.shape)
print('Y validation')
print(Y_validation.shape)'''


# In[ ]:

from sklearn import metrics
seed = 7
scoring = 'accuracy'
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#for n in range(5,6):
#    models.append(('KNN', KNeighborsClassifier(n_neighbors=n,n_jobs=-1)))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
i=5
#for name, model in models:
	#kfold = model_selection.KFold(n_splits=10, random_state=seed)
	#cv_results = model_selection.cross_val_score(model, X_train, Y_train,cv=10,scoring=scoring)
	#results.append(cv_results)
	#names.append(name)
    #pickle.dump(model, open(filename, 'wb'))
	#msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	#print(msg)
print(i)
model=KNeighborsClassifier(n_jobs=-1)
model.fit(X,Y)
#pre=model.predict(X_validation)
#print(name)
#print(metrics.accuracy_score(Y_validation,pre))
res=model.predict(test)
        #print(str(i+1)+str(res[i]))
    #print(metrics.classification_report(Y_validation, pre))
    #print(metrics.confusion_matrix(Y_validation, pre))'''


# In[ ]:

with open('result.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n',delimiter=",",quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["imageid","label"])
    for i in range(0,len(res)):
        writer.writerow([i+1,res[i]])


# In[ ]:

import os
curpath = os.path.abspath(os.curdir)
print(curpath)


# In[ ]:




# In[ ]:



