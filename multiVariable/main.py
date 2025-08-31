import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Importing the data
data = pd.read_csv("multi.csv")
size = Variable(torch.tensor(data[["size"]].to_numpy()))
location_score = Variable(torch.tensor(data[["location_score"]].to_numpy()))
age = Variable(torch.tensor(data[["age"]].to_numpy()))
price = Variable(torch.tensor(data[["price"]].to_numpy()))
rental_yield = Variable(torch.tensor(data[["rental_yield"]].to_numpy()))

# Plotting the original data
plt.scatter(size, location_score, age, price, rental_yield)

# LinearRegression
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(3,2, dtype=torch.float64)

    def forward(self, x,y,z):
        return self.linear(x,y,z)


# Executing
model = LinearRegression()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

loss_list = []

for epoch in range(10):
    # 1. predict current
    price, rental_yield = model(size,age,location_score)
    # 2. compute loss from original
    loss = criterion(price, salary)
    # 3. Calculating gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    print(f"Epoch: {epoch}, loss: {loss_list[-1]}")

m, c = list(model.parameters())

experience = yearsExperience
Y = m * experience + c
plt.plot(experience, Y.detach())
plt.show()
