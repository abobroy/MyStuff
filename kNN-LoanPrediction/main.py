# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:29:00 2017

@author: tony
"""
import pandas as pd
import numpy as np
pd.set_option('max_columns', 180)
pd.set_option('max_colwidth', 5000)

import matplotlib.pyplot as plt
import seaborn as sns

# Import kNN machine learning algo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

from modCleanInputData import cleanInputData
#from modBackfillInputData import backfillInputData
import modBackfillInputData

from importlib import reload

#reload(modCleanInputData)
reload(modBackfillInputData)

#import IPython
#print(IPython.sys_info())

#%matplotlib inline
#plt.rcParams['figure.figsize'] = (12,8)

def kNNPredictor(trainData):
    print("########################################")
    print("### Create test/validate data sets from data")
    # create design matrix X and target vector y
    X = np.array(trainData.loc[:, trainData.columns != 'Loan_Status']) 	# end index is exclusive
    y = np.array(trainData["Loan_Status"]) 	# another way of indexing a pandas df
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    print("########################################")
    print("### Run 10-fold cross-validation to find optimal number of neighbors")
    # creating odd list of K for KNN from 1 to 50
    neighbors = np.arange(1, 50, 2)
    # empty list that will hold cv scores
    cv_scores = []
    
    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
        #print(scores.mean())
    
    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]
    
    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    cv_score = max(cv_scores)
    print("########################################")
    print("The optimal number of neighbors is {0}".format(optimal_k))
    
    # plot misclassification error vs k
    #plt.plot(neighbors, MSE)
    #plt.xlabel('Number of Neighbors K')
    #plt.ylabel('Misclassification Error')
    #plt.show()
    
    # Run Sklearn algo with optimal number of neighbors
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    # fitting the model
    knn.fit(X_train, y_train)    
    # predict the response
    pred = knn.predict(X_test)
    # evaluate accuracy
    print("Cross-Validation Score {0:4.2%}, kNN accuracy={1:4.2%}".format(cv_score, accuracy_score(y_test, pred)))

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
####################################################################################
    
filePath = "C:/Users/tony/Miniconda3/envs/tonyEnv/kNN-LoanPrediction/"
trainFileName = "train_data.csv"
fileDataDict = "data_dictionary.csv"
filteredFileName = "filtered_data.csv"

# Call to clean data
#trainData = cleanInputData(filePath, trainFileName, fileDataDict, filteredFileName)
trainData = modBackfillInputData.backfillInputData(filePath, trainFileName, fileDataDict, filteredFileName)

kNNPredictor(trainData)

outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, trainData, predictor_var,outcome_var)

#We can try different combination of variables:
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, trainData, predictor_var,outcome_var)

model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, trainData, predictor_var,outcome_var)

#We can try different combination of variables:
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, trainData, predictor_var,outcome_var)

model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','ApplicantIncome_log', 'CoapplicantIncome_log']
classification_model(model, trainData, predictor_var,outcome_var)

#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['ApplicantIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, trainData, predictor_var,outcome_var)