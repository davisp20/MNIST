# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:06:14 2020

@author: pdavi
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE                   # For Oversampling
from sklearn.neural_network import *

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#print(train.describe())
X = train[train.columns[1:]].values
y = np.vstack(train['label'].values)

# print('\n')
# print('X and y Input Data:   ', X.shape, y.shape)


X_train, X_test2, y_train, y_test2 = train_test_split(X, y, test_size=0.3,
                                                                         random_state=42)

# print('Training Set Shape:   ', X_train_original.shape, y_train_original.shape)

X_val, X_test, y_val, y_test = train_test_split(X_test2, y_test2, test_size=0.33,random_state=42)

# print('Validation Set Shape: ', X_val.shape,y_val.shape)
# print('Test Set Shape:       ', X_test.shape, y_test.shape)

#doOversampling = True

# if doOversampling:
# # Apply regular SMOTE
#     sm = SMOTE()
#     X_train, y_train = sm.fit_sample(X_train_original, y_train_original)
# print('Training Set Shape after oversampling:   ', X_train.shape, y_train.shape)
# print(pd.crosstab(y_train,y_train))
# else:
#     X_train = X_train_original
#     y_train = y_train_original


#NN Classifier
MLPClassifier(solver = 'adam', activation='relu', alpha=1e-04,
        batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
        epsilon=1e-08, hidden_layer_sizes=(100), learning_rate='adaptive',
        learning_rate_init=0.005, max_iter=2500, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
        tol=1e-05, validation_fraction=0.1, verbose=True,
        warm_start=True)

clf_MLP = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(64))

clf_MLP.fit(X_train, y_train)
y_pred_MLP = clf_MLP.predict(X_val)

print('       Accuracy of Model       ')
print('--------------------------------')
print('Neural Network          '+"{:.2f}".format(accuracy_score(y_val, y_pred_MLP)*100)+'%')

#Confusion Matrices
# print('Neural Network  ')
# cm_MLP = confusion_matrix(y_val,y_pred_MLP)
# print(cm_MLP)
# print('\n')

# y_test = y_test.reshape(-1)
# y_train_original = y_train.reshape(-1)

# y_pred_train_MLP = clf_MLP.predict(X_train)
# y_pred_test_MLP = clf_MLP.predict(X_test)
# cm_MLP_train = confusion_matrix(y_train_original,y_pred_train_MLP)
# cm_MLP_test = confusion_matrix(y_test,y_pred_test_MLP)

# print('Neural Network Classification Matrix ')
# print('Training')
# print(cm_MLP_train)
# print('Validation')
# print(cm_MLP)
# print('Test')
# print(cm_MLP_test)
# print("\n")

ypred = clf_MLP.predict(test)
import csv
changes = []
for i in range(len(ypred)):
    changes.append((i+1,ypred[i]))  
                                                                   
outfile=open('digits1.csv','w', newline='')
writer=csv.writer(outfile)
writer.writerow(["ImageId", "Label"])
writer.writerows(changes)