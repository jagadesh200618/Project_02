import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Importing the data
data = pd.read_csv("dataset.csv")
yearsExperience = Variable(torch.tensor(data[["YearsExperience"]].to_numpy()))
salary = Variable(torch.tensor(data[["Salary"]].to_numpy()))

# Plotting the original data
plt.scatter(yearsExperience, salary)

# LinearRegression
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1,1, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x)


# Executing
model = LinearRegression()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

loss_list = []

for epoch in range(10):
    # 1. predict current salary
    salary_pred = model(yearsExperience)
    # 2. compute loss from original
    loss = criterion(salary_pred, salary)
    # 3. Calculating gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    print(f"Epoch: {epoch}, loss: {loss_list[-1]}")

parameter = model.parameters()

m = next(parameter).detach().numpy()
c = next(parameter).detach().numpy()

experience = yearsExperience.detach().numpy()
Y = m * experience + c
plt.plot(experience, Y)
plt.show()
