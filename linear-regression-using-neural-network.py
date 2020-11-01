import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# hyper parameter
input_size = 1
output_size = 1
num_epochs = 600
lr = 0.001

# dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
    [9.779], [6.182], [7.59], [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
    [3.366], [2.596], [2.53], [1.221], [2.827], [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.linear(x)
        return output
        

# model
model = LinearRegression(input_size, output_size)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# train
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # forward + backward + optimize
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1)%5 == 0:
        print('Epoch [%d/%d], Loss: %s'%(epoch+1, num_epochs, loss))


print(model.state_dict())

# plot the graph
predicted = model(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()


