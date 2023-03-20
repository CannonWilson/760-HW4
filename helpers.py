import numpy as np

def sig(x):
    return 1.0/(1.0 + np.exp(-x))

def sig_deriv(x):
    return (1.0 - sig(x)) * sig(x)

def relu(x):
    return np.maximum(0,x)

def relu_deriv(x):
    return np.where(x >= 0, 1, 0)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def softmax_deriv(y_hat,y): 
    return y_hat - y

def logloss(y, y_hat):
    return -np.sum(y*np.log(softmax(y_hat)))


def logloss_deriv(y, y_hat):
    # derivative of above function using derivative of
    # sum is equiv. to sum of derivatives
    return -y/y_hat