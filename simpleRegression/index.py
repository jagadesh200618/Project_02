import pandas as pd 
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable 

data = pd.read_csv('data.csv')
yearsexperience = Variable(torch.tensor(data['YearsExperience'].to_numpy()))
salary = Variable(torch.tensor(data['Salary'].to_numpy()))

plt.scatter(yearsexperience, salary)

class LinearRegressionModel:
    def __init__(self):
        self(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x)
    
        
print(salary)
 
