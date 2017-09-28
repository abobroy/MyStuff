# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 21:14:29 2017

@author: tony
"""
import pandas as pd
#import numpy as np
pd.set_option('max_columns', 180)
pd.set_option('max_colwidth', 5000)
from sklearn.preprocessing import LabelEncoder

#import matplotlib.pyplot as plt
#import seaborn as sns
"""
Data set column descriptions
===========================================
Variable          Description
Loan_ID           Unique Loan ID
Gender            Male/ Female
Married           Applicant married (Y/N)
Dependents        Number of dependents
Education         Applicant Education (Graduate/ Under Graduate)
Self_Employed     Self employed (Y/N)
ApplicantIncome   Applicant income
CoapplicantIncome Coapplicant income
LoanAmount        Loan amount in thousands
Loan_Amount_Term  Term of loan in months
Credit_History    credit history meets guidelines
Property_Area     Urban/ Semi Urban/ Rural
Loan_Status       Loan approved (Y/N)
"""

def cleanInputData(inputDataFilePath, trainFileName, dataDictName, filteredFileName):
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
    
    # Visualize LoanAmount data, Histogram with 20 bins
    #loan_data['LoanAmount'].plot.hist(alpha=0.5,bins=20,label='Loan Amounts');
    #loan_data['ApplicantIncome'].hist(bins=50)
    
    # Visualize LoanAmount data
    #fig, axs = plt.subplots(1,2,figsize=(14,7))
    #sns.countplot(x='Loan_Status',data=loan_data,ax=axs[0])
    #axs[0].set_title("Frequency of each Loan Status")
    #loan_data.Loan_Status.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
    #axs[1].set_title("Percentage of each Loan status")
    #plt.show()
    
    # Look for columns with that have few unique values
    print("###############################################")
    print("### Print all columns with 4 or less unique values because they may not help with prediction")
    for col in loan_data.columns:
        if (len(loan_data[col].unique()) < 4):
            print(loan_data[col].value_counts())
            print()
            
    print("###############################################")
    print("### Number of NULL values in each column")        
    null_counts = loan_data.isnull().sum()
    print(null_counts)
    
    print("###############################################")
    print("### Drop NAs")     
    loan_data = loan_data.dropna()
    
    # check Load_status column possible values
    #print("### BEFORE check for Loan_Status values being Y/N")
    #print(loan_data["Loan_Status"].value_counts())
    # Data clean up, just in case get rid of any rows that are not "Y/N" in Loan_status column    
    #loan_data = loan_data[(loan_data["Loan_Status"] == "Y") | (loan_data["Loan_Status"] == "N")]
    # Replace textual column with numbers    
    #mapping_dict = {"Loan_Status":{ "Y": 1, "N": 0}}
    #loan_data = loan_data.replace(mapping_dict)

#    print("###############################################")
#    print("### Convert 'Dependents' column to numeric")  
#    loan_data.loc[loan_data["Dependents"] == "3+", "Dependents"] = float(3)
#    loan_data["Dependents"] = loan_data["Dependents"].apply(pd.to_numeric)
    
    print("###############################################")
    print("### Show categorical columns")      
    object_columns_df = loan_data.select_dtypes(include=['object'])
    print(object_columns_df.iloc[0])
#    mapping_dict = {
#            "Education":{ "Graduate": 1, "Not Graduate": 0},
#            "Gender":{ "Male": 1, "Female": 0},
#            "Married":{ "Yes": 1, "No": 0},
#            "Self_Employed":{ "Yes": 1, "No": 0},
#            "Property_Area":{ "Rural": 0, "Semiurban": 1, "Urban": 2}
#            }
#    loan_data = loan_data.replace(mapping_dict)
    print("###############################################")
    print("### Encode categorical columns to numeric")      
    var_mod = ['Gender','Married', 'Dependents', 'Education','Self_Employed',
               'Property_Area', 'Loan_Status']
    le = LabelEncoder()
    for i in var_mod:
        loan_data[i] = le.fit_transform(loan_data[i])
    print(loan_data.dtypes)

    # drop columns that leak data from the future or have no effect on the loan approval
    print("###############################################")
    print("### Drop columns that leak data from the future or have no effect on the loan approval")                
    drop_list = ["Loan_ID", "Dependents", "Education", "Property_Area"]
    loan_data = loan_data.drop(drop_list,axis=1) 
    
    print("###############################################")
    print("### Check column data types and counts")      
    print(loan_data.info())
    
    print("###############################################")
    print("### Save final version of filtered data to CSV") 
    filteredName = inputDataFilePath + filteredFileName
    loan_data.to_csv(filteredName,index=False)          
    
    return loan_data