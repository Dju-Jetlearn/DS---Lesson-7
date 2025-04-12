import matplotlib.pyplot as matpat
import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/diego/JetLearn/Data Science/Lesson 7/Data.csv")
print(data.info())

# Steps to be followed in a Machine Leanring Project - 
#    a) Gathering the data and load the data into the program
#    b) Data Preprocessing - prepare the data for further machine learning projects
#    c) Data Analysis - Analyse the data so that you can define your input and output columns and drop the rest
#    d) Define Input and Output - Create separate dataframes for input - output 
#    e) Perform the test-train split - Split up the data into testing and training data , Explain the use of training and testing data
#    f) Select the machine learning algorithm perform the training of the model.
#    g) Perform predictions and compare the predictions with actual result to calculate accuracy.

# Data Analysis
x = data.iloc[:, :-1].values # picks all columns except the last one
y = data.iloc[:, -1].values # picks last column

print("Features :\n", x)
print("Target :\n", y)

# Data Preproccessing

# 1.Replacing Null values with the means of their columns

#simple imputer -> replaces missing values with other values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,  strategy = 'mean')

# Transform the data for the entire dataset (age and salary alone)

x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
print("After Imputing :\n", x)

# 2.Encoding categorical data to numerical value (ex : replacing country names with 0, 1 or 2)

# Replacing 'Purchased' value with 1s and 0s

from sklearn.preprocessing import LabelEncoder

# Yes = 1, No = 0
le = LabelEncoder()
y = le.fit_transform(y)
print("Label Encoder :\n", y)
