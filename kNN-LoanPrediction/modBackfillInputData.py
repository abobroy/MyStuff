# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:55:57 2017

@author: tony
"""
import pandas as pd
import numpy as np
pd.set_option('max_columns', 180)
pd.set_option('max_colwidth', 5000)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

def backfillInputData(inputDataFilePath, trainFileName, dataDictName, filteredFileName):
    """
    cleanInputData
    """
    fileName = inputDataFilePath + trainFileName
    loan_data = pd.read_csv(fileName, header=0) 
    #half_count = len(loan_data) / 2
    #loan_data = loan_data.dropna(thresh=half_count,axis=1) # Drop any column with more than 50% missing values
    #loan_data = loan_data.drop(['url','desc'],axis=1)      # These columns are not useful for our purposes
    print("###############################################")
    print("### Check column data types and counts")      
    print("Loaded test data set (rows, columns)={0}:".format(loan_data.shape))
    print(loan_data.info())
    
    # Load column descriptions
    dataDict = inputDataFilePath + dataDictName
    data_dictionary = pd.read_csv(dataDict, header=0) # Loading in the data dictionary
    #print(data_dictionary)
    data_dictionary = data_dictionary.rename(columns={'Variable': 'name', 'Description': 'description'})
    
    # Join loan dataFrame with column dictionary and the first data row to inspect data visually
    loans_dtypes = pd.DataFrame(loan_data.dtypes,columns=['dtypes'])
    loans_dtypes = loans_dtypes.reset_index()
    loans_dtypes['name'] = loans_dtypes['index']
    loans_dtypes = loans_dtypes[['name','dtypes']]
    
    loans_dtypes['first value'] = loan_data.loc[0].values
    preview = loans_dtypes.merge(data_dictionary, on='name',how='left')
    print("###############################################")
    print("### Column dictionary with the first row of data")
    print(preview[0:25])   
    
    print("###############################################")
    print("### Number of NULL values in each column")        
    print(loan_data.isnull().sum())

    ### [NOTE] Tried using Imputer on Self_Employed column, but it only works on numeric data    
    ###np_values = loan_data["Self_Employed"].values
    ###imputer = Imputer(missing_values='NaN', strategy="most_frequent", axis=0)
    ###loan_data["Self_Employed"] = imputer.fit_transform(np_values)
    print("###############################################")
    print("### Replace Self_Employed and Gender NA with most frequent value")
    frequent_value = loan_data['Self_Employed'].value_counts().idxmax()
    loan_data['Self_Employed'].fillna(frequent_value,inplace=True)
    frequent_value = loan_data['Gender'].value_counts().idxmax()
    loan_data['Gender'].fillna(frequent_value,inplace=True)    
    frequent_value = loan_data['Married'].value_counts().idxmax()
    loan_data['Married'].fillna(frequent_value,inplace=True)      
    frequent_value = loan_data['Dependents'].value_counts().idxmax()
    loan_data['Dependents'].fillna(frequent_value,inplace=True)  
    
    print("###############################################")
    print("### Replace Loan_Amount_Term NA with the median of this column")
    Loan_Amount_Term_median = loan_data["Loan_Amount_Term"].median()
    print("Loan_Amount_Term_median = {0:4.2f}".format(Loan_Amount_Term_median))
    loan_data['Loan_Amount_Term'].fillna(frequent_value,inplace=True)    
    
    print("###############################################")
    print("### Pivot table of median LoanAmount by Self_Employed/Education")  
    #loan_data['LoanAmount'].hist(bins=50)
    #loan_data.boxplot(column='LoanAmount')
    # Fill missing values in LoanAmount column based on the median of Self_Employed/Education cohorts
    # Calculate a pivot table of median LoanAmounts by Self_Employed by Education cohorts
    pivot = loan_data.pivot_table(index='Self_Employed', columns='Education', 
                                  values='LoanAmount', aggfunc=np.median)
    #print(pivot)
    # Define function to return value of this pivot table
    def fage(x):
        return pivot.loc[x['Self_Employed'],x['Education']]
    # Fill LoanAmount NA with median values from the pivot table
    loan_data['LoanAmount'].fillna(loan_data[loan_data['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
    
    print("###############################################")
    print("### Number of NULL values in each column")        
    print(loan_data.isnull().sum())

    print("###############################################")
    print("### Convert LoanAmount to log to change to Normal distribution")     
    loan_data["LoanAmount_log"] = pd.Series(np.log(loan_data['LoanAmount']), index=loan_data.index)
    col_list = ['LoanAmount', 'LoanAmount_log']
    #loan_data[col_list].hist(alpha=0.5, bins=20)
    
    print("###############################################")
    print("### Convert ApplicantIncome to log to change to Normal distribution")     
    loan_data["ApplicantIncome_log"] = pd.Series(np.log(loan_data['ApplicantIncome']), index=loan_data.index)
    col_list = ['ApplicantIncome', 'ApplicantIncome_log']
    #loan_data[col_list].hist(alpha=0.5, bins=20)

    print("###############################################")
    print("### Convert CoapplicantIncome to log to change to Normal distribution")  
    # Calculate mean of 'CoapplicantIncome' column excluding 0 values
    #CoapplicantIncome_median = loan_data[loan_data.CoapplicantIncome>0, "CoapplicantIncome"].median()
    #print("CoapplicantIncome_median = {0:4.2f}".format(CoapplicantIncome_median))
    loan_data.loc[loan_data.CoapplicantIncome==0, 'CoapplicantIncome'] = 0.1
    loan_data["CoapplicantIncome_log"] = pd.Series(np.log(loan_data['CoapplicantIncome']), index=loan_data.index)
    #col_list = ['CoapplicantIncome', 'CoapplicantIncome_log']
    #loan_data[col_list].hist(bins=20)

    print("###############################################")
    print("### Encode categorical columns to numeric")      
    var_mod = ['Married', 'Dependents', 'Education', 'Gender', 'Self_Employed', 
               'Property_Area', 'Loan_Status']
    le = LabelEncoder()
    for i in var_mod:
        loan_data[i] = le.fit_transform(loan_data[i])
    print(loan_data.dtypes)        
    
    print("###############################################")
    print("### Fill NAs in Credit History based on Loan_Status values")     
    #loan_data['Credit_History'].fillna(loan_data['Loan_Status'], inplace=True)
    print("### Drop NAs")     
    loan_data = loan_data.dropna()
    
    print("###############################################")
    print("### Frequency for 'Credit_History' and probality of getting loan for each Credit History class:")
    Credit_History_Pivot = loan_data.pivot_table(index=['Credit_History'], 
           values=['Loan_Status'], aggfunc={len, np.mean})    
    print(Credit_History_Pivot)
    # Get 'Len' column from the pivot table data frame
    bar_data = Credit_History_Pivot.iloc[:, 0].values
    #print("Type {0}".format(type(bar_data)))  
    idx = np.arange(2)
    plt.bar(idx, Credit_History_Pivot.iloc[:, 0].values, color="blue", width=0.3)
    plt.title("Applicants by Credit_History")
    plt.ylabel("Count of applicants")
    plt.xlabel("Credit_History")    
    plt.xticks(idx, ('0', '1'))
    plt.show()
    
    plt.bar(idx, Credit_History_Pivot.iloc[:, 1].values, color="blue", width=0.3)
    plt.title("Probability of loan by credit history")    
    plt.ylabel("Probability of getting loan")
    plt.xlabel("Credit_History")    
    plt.xticks(idx, ('0', '1'))
    plt.show()    
    
    # drop columns that leak data from the future or have no effect on the loan approval
    print("###############################################")
    print("### Drop columns that leak data from the future, have no effect on the loan approval, or were LOGed")                
    # "Dependents", "Education", "Property_Area", "LoanAmount", 
    drop_list = ["Loan_ID", "LoanAmount", "ApplicantIncome", "CoapplicantIncome"]    
    loan_data = loan_data.drop(drop_list,axis=1)     
    
    print("###############################################")
    print("### Check column data types and counts")      
    #print(loan_data['Credit_History'].describe())          
    print(loan_data.info())
        
    print("###############################################")
    print("### Save final version of filtered data to CSV") 
    filteredName = inputDataFilePath + filteredFileName
    loan_data.to_csv(filteredName,index=False)          
    
    return loan_data    