import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Importing the data
data = pd.read_csv("multi.csv")
input = torch.tensor(data[["size", "location_score", "age"]].to_numpy())
output = torch.tensor(data[["price", "rental_yield"]].to_numpy())


# LinearRegression
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(3,2, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x)


# Executing
model = LinearRegression()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

loss_list = []

for epoch in range(10):
    # 1. predict current
    prediction = model(input)
    # 2. compute loss from original
    loss = criterion(output, prediction)
    # 3. Calculating gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    print(f"Epoch: {epoch}, loss: {loss_list[-1]}")

m = list(model.parameters())
print(m)

