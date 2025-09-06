import pandas as pd
import numpy as np


data = pd.read_csv('data.csv')
yearsexperience = data['YearsExperience']
salary = data['Salary']


#print(yearsexperience, salary)

S_row = data.iloc[10:20]
F_row = S_row[S_row['Salary']>80000]
print(F_row)
    
    
