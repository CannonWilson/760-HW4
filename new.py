import numpy as np
import torchvision
import torch
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from helpers import sig, sig_deriv, relu, relu_deriv, softmax, softmax_deriv, logloss, logloss_deriv


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

NUM_CLASSES = 10
np.random.seed(0)

# Initial Setup   
print('USING 0 WEIGHTS') 
w1=np.zeros((300, 784))# np.random.randn(300,784) # np.random.uniform(-1, 1, (300, 784))
b1=np.zeros((300,1)) # np.random.randn(300,1) # np.random.uniform(-1, 1, (300, 1))
z1=None
a1=None
w2=np.zeros((200, 300))# np.random.randn(200,300) # np.random.uniform(-1, 1, (200, 300)) 
b2=np.zeros((200, 1))# np.random.randn(200,1) # np.random.uniform(-1, 1, (200, 1))
z2=None
a2=None
w3=np.zeros((10,200))# np.random.randn(10,200) # np.random.uniform(-1, 1, (10, 200))
b3=np.zeros((10, 1))# np.random.randn(10,1) # np.random.uniform(-1, 1, (10, 1))
z3=None
y_hat=None

def forward(img):
    global w1,b1,z1,a1,w2,b2,z2,a2,w3,b3,z3,y_hat

    z1 = w1@img + b1
    a1 = sig(z1)
    z2 = w2@a1 + b2
    a2 = sig(z2)
    z3 = w3@a2 + b3
    y_hat = softmax(z3).reshape(-1, 1)


def backward(img, y, lr):
    global w1,b1,z1,a1,w2,b2,z2,a2,w3,b3,z3,y_hat
    
    dz3 = softmax_deriv(y_hat, y) # logloss_deriv(y, z3) 
    dw3 = dz3 @ a2.T
    db3 = np.copy(dz3) # np.sum(dz3, axis=1, keepdims=True)
    dz2 = sig_deriv(z2) * (w3.T @ dz3)
    dw2 = db2 = dz2 @ a1.T
    db2 = np.copy(dz2) # np.sum(dz2, axis=1, keepdims=True)
    dz1 = sig_deriv(z1) * (w2.T @ dz2)
    dw1 = db1 = dz1 @ img.T
    db1 = np.copy(dz1) # np.sum(dz1, axis=1, keepdims=True)

    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1
    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2
    w3 = w3 - lr * dw3
    b3 = b3 - lr * db3


def train(epochs, lr=0.001):
    global w1,b1,z1,a1,w2,b2,z2,a2,w3,b3,z3,y_hat

    losses_list = []
    test_accs = []

    for epoch in range(epochs):
        
        train_data_iter = iter(train_data_loader)
        epoch_loss = 0
        acc_count = 0

        for img, lbl in train_data_iter: # img has shape [BS, 1, 28, 28]
            
            # Stop if any nan values occur
            assert not np.isnan(np.sum(w1)) and not np.isnan(np.sum(b1)) and \
                not np.isnan(np.sum(w2)) and not np.isnan(np.sum(b2)) and \
                not np.isnan(np.sum(w3)) and not np.isnan(np.sum(b3))

            # Format img and lbl
            img = torch.reshape(img, (784, 1))
            img = img.numpy()
            one_hot_y = np.zeros((NUM_CLASSES, 1))
            one_hot_y[lbl] = 1

            forward(img)
            epoch_loss += logloss(one_hot_y, y_hat)
            if np.argmax(y_hat) == lbl: acc_count += 1
            backward(img, one_hot_y, lr)

        print(f'Avg loss for epoch {epoch + 1}: {epoch_loss/len(train_data_iter)}')
        print(f'Acc for epoch: {acc_count/len(train_data_iter)}')
        losses_list.append(epoch_loss)

        test_accs.append(test())


    # Show learning curve   
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.title('Learning Curve w/ 0 weight init')# plt.title('Learning Curve')
    plt.plot(range(1, len(losses_list)+1), losses_list)
    plt.show()

    # Show test accuracy
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy w/ 0 weight init')# plt.title('Test Accuracy') 
    plt.plot(range(1, len(losses_list)+1), test_accs)
    plt.show()

def test():
    global w1,b1,z1,a1,w2,b2,z2,a2,w3,b3,z3,y_hat

    test_iter = iter(test_data_loader)
    acc_count = 0

    for img, lbl in test_iter: # img has shape [BS, 1, 28, 28]
            
        # Stop if any nan values occur
        assert not np.isnan(np.sum(w1)) and not np.isnan(np.sum(b1)) and \
            not np.isnan(np.sum(w2)) and not np.isnan(np.sum(b2)) and \
            not np.isnan(np.sum(w3)) and not np.isnan(np.sum(b3))

        # Format img and lbl
        img = torch.reshape(img, (784, 1))
        img = img.numpy()
        one_hot_y = np.zeros((NUM_CLASSES, 1))
        one_hot_y[lbl] = 1
        forward(img)
        if np.argmax(y_hat) == lbl: acc_count += 1
    acc = acc_count / len(test_iter)
    print('Epoch accuracy: ', acc)
    return acc

if __name__ == '__main__':
    train(epochs=10)
    test()