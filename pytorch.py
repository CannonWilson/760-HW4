import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self, lr):
        super(Net, self).__init__()
        self.lr = lr
        self.l1 = nn.Linear(784, 300)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(300, 200)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(200, 10)
        self.a3 = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)
        return x
    

MNIST_train_data = torchvision.datasets.MNIST('./mnist', 
                                             train=True,
                                             download=True,
                                             transform=ToTensor())
MNIST_test_data = torchvision.datasets.MNIST('./mnist', 
                                             train=False,
                                             download=True,
                                             transform=ToTensor())
train_data_loader = torch.utils.data.DataLoader(MNIST_train_data,
                                                batch_size=1,
                                                shuffle=True)
test_data_loader = torch.utils.data.DataLoader(MNIST_test_data,
                                               batch_size=1,
                                               shuffle=True)

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Initialize to 0
        # m.weight.data.fill_(0)
        # m.bias.data.fill_(0)
        # print('weights and biases initialized to 0')

        # Initialize to [-1, 1]
        m.weight.data.trunc_normal_(a=-1, b=1)
        m.bias.data.trunc_normal_(a=-1, b=1)
        print("WEIGHT DATA:")
        print(m.weight.data)
        print('weights and biases initialized to [-1, 1]')
        


EPOCHS = 10
model = Net(lr=0.001)
model.apply(init_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001)

training_losses = []
test_accuracies = []

# Train
model.train()
for epoch in range(EPOCHS):

    epoch_loss = 0
    train_acc = 0

    for img, lbl in iter(train_data_loader):
        optimizer.zero_grad()
        img = torch.reshape(img, (1, 784))
        one_hot_y = torch.zeros((1, 10))
        one_hot_y[0][lbl] = 1
        out = model.forward(img)
        pred = torch.argmax(out[0]).item()
        if pred == lbl: train_acc += 1
        loss = F.cross_entropy(out, one_hot_y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    training_losses.append(epoch_loss)
    print(f'Finished training epoch {epoch + 1} with loss: {loss}')
    print(f'Finished training epoch {epoch + 1} with acc: {train_acc/len(iter(train_data_loader))}')

    # Test
    acc_count = 0
    for img, lbl in iter(test_data_loader):
        img = torch.reshape(img, (1, 784))
        out = model.forward(img)
        pred = torch.argmax(out[0]).item()
        if pred == lbl: acc_count += 1
    acc = acc_count / len(iter(test_data_loader))
    print(f'Finished testing for epoch {epoch + 1} with acc: {acc}')
    test_accuracies.append(acc)

 # Show learning curve   
plt.xlabel('Epoch') 
plt.ylabel('Loss')
plt.title('Learning Curve w/ weights initialized to 0')
plt.plot(range(1, len(training_losses)+1), training_losses)
plt.show()

# Show test accuracy
plt.xlabel('Epoch') 
plt.ylabel('Accuracy')
plt.title('Test Accuracy w/ weights initialized to 0')
plt.plot(range(1, len(test_accuracies)+1), test_accuracies)
plt.show()
