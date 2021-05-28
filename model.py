import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Election_2019.csv')

# renaming column names
df = df.rename(columns={'CRIMINAL\nCASES': 'CRIMINAL_CASES', 'GENERAL\nVOTES': 'GENERAL_VOTES', 'POSTAL\nVOTES': 'POSTAL_VOTES', 'TOTAL\nVOTES': 'TOTAL_VOTES',
                        'OVER TOTAL ELECTORS \nIN CONSTITUENCY': 'OVER_TOTAL_ELECTORS_IN_CONSTITUENCY', 'OVER TOTAL VOTES POLLED \nIN CONSTITUENCY': 'OVER_TOTAL_VOTES_POLLED_IN_CONSTITUENCY', 'TOTAL ELECTORS': 'TOTAL_ELECTORS'})

# drop rows with NA values
df = df[df['GENDER'].notna()]

# replace Nil values with 0
df['ASSETS'] = df['ASSETS'].replace(['Nil', '`', 'Not Available'], '0')
df['LIABILITIES'] = df['LIABILITIES'].replace(
    ['NIL', '`', 'Not Available'], '0')
df['CRIMINAL_CASES'] = df['CRIMINAL_CASES'].replace(['Not Available'], '0')

# clean ASSETS and LIABILITIES column values
df['ASSETS'] = df['ASSETS'].map(lambda x: x.lstrip(
    'Rs ').split('\n')[0].replace(',', ''))
df['LIABILITIES'] = df['LIABILITIES'].map(
    lambda x: x.lstrip('Rs ').split('\n')[0].replace(',', ''))

# convert ASSETS, LIABILITIES and CRIMINAL_CASES column values into numeric
df['ASSETS'] = df['ASSETS'].astype(str).astype(float)
df['LIABILITIES'] = df['LIABILITIES'].astype(str).astype(float)
df['CRIMINAL_CASES'] = df['CRIMINAL_CASES'].astype(str).astype(int)

# reorder columns
cols = df.columns.tolist()
cols = cols[0:3] + cols[4:] + cols[3:4]
df = df[cols]

df_new = df[['GENERAL_VOTES', 'POSTAL_VOTES', 'TOTAL_VOTES',
             'OVER_TOTAL_ELECTORS_IN_CONSTITUENCY', 'OVER_TOTAL_VOTES_POLLED_IN_CONSTITUENCY', 'WINNER']]

#Splitting the data into test and train set

X=df_new.drop(['WINNER'],axis=1)
y=df_new['WINNER']


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=42,test_size=0.20)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(Xtrain,ytrain)

import pickle
# open a file, where you ant to store the data
file = open('main_proj.pkl', 'wb')

# dump information to that file
pickle.dump(rf, file)