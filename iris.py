#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:20:43 2019

@author: mohammedgouse
"""

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
warnings.simplefilter(action='ignore', category=FutureWarning)
#reading the data and creating a dataframe

iris = pd.read_csv('Iris.csv')
iris = iris.drop(['Id'],axis = 1)

# Data visualization

sns.countplot(x = "Species" , data = iris)

fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa') 
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

#histogram
iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

sns.scatterplot(x = iris.SepalLengthCm , y = iris.SepalWidthCm)
sns.scatterplot(x = iris.PetalLengthCm , y = iris.PetalWidthCm)


sns.heatmap(iris.corr(), annot = True, cmap='cubehelix_r') #heatmap


# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn import metrics #for checking the model accuracy


train , test = train_test_split(iris, test_size = 0.3)


X_train = train.drop(["Species"] , axis = 1)
Y_train = train.Species

X_test = test.drop(["Species"] , axis = 1)
Y_test = test.Species

# Logistic Regression
Logistic_regressor = LogisticRegression()
Logistic_regressor.fit(X_train, Y_train)
Logistic_prediction = Logistic_regressor.predict(X_test)

# SVM Classifcation
from sklearn import svm

svm_classifier = svm.SVC()
svm_classifier.fit(X_train, Y_train)
svm_prediction = svm_classifier.predict(X_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT_classifier = DecisionTreeClassifier()
DT_classifier.fit(X_train, Y_train)
DT_prediction = DT_classifier.predict(X_test)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators = 10 , criterion= 'entropy')
rf_classifier.fit(X_train, Y_train)
rf_prediction = rf_classifier.predict(X_test)

#Metrics
from sklearn.metrics import confusion_matrix
Logistic_cm = confusion_matrix(Y_test, Logistic_prediction)
Logistic_accuracy = metrics.accuracy_score(Y_test, Logistic_prediction)

DT_cm = confusion_matrix(Y_test, DT_prediction)
DT_accuracy = metrics.accuracy_score(Y_test, DT_prediction)

svm_cm = confusion_matrix(Y_test, svm_prediction)
svm_accuracy= metrics.accuracy_score(Y_test, svm_prediction)

rf_cm = confusion_matrix(Y_test, rf_prediction)
rf_accuracy= metrics.accuracy_score(Y_test, rf_prediction)

from sklearn.model_selection import cross_val_score
Logistic_accuracies = cross_val_score(estimator = Logistic_regressor, X = X_train, y = Y_train, cv = 10)
Logistic_accuracies.mean()
Logistic_accuracies.std()

DT_accuracies = cross_val_score(estimator = DT_classifier, X = X_train, y = Y_train, cv = 10)
DT_accuracies.mean()
DT_accuracies.std()

svm_accuracies = cross_val_score(estimator = svm_classifier, X = X_train, y = Y_train, cv = 10)
svm_accuracies.mean()
svm_accuracies.std()

rf_accuracies = cross_val_score(estimator = rf_classifier, X = X_train, y = Y_train, cv = 10)
rf_accuracies.mean()
rf_accuracies.std()


print ("\n Confusion Matrix - Accuracy of each model")
print ("\n Logistic Regression: ", Logistic_accuracy, " \n Decision Tree: ", DT_accuracy, "\n Support Vector Machine: ", svm_accuracy, "\n Random Forest: ", rf_accuracy)                              

print ("\n Cross Validation Accuracies")
print("\n Logistic Regression: ", Logistic_accuracies.mean(), " \n Decision Tree: ", DT_accuracies.mean(), "\n Support Vector Machine: ", svm_accuracies.mean(), "\n Random Forest: ", rf_accuracies.mean())    



