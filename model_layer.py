import numpy as np
import sys
from optimizers import *

np.random.seed(2333)


class Linear(object):
    def __init__(self, in_dim, out_dim, name = 'linear', lr = 0.01, optimizer_type = 'adam'):
        print(name+' init: {}->{}'.format(in_dim, out_dim))
        self.name = name
        self.W = np.random.randn(in_dim, out_dim)
        self.b = np.random.randn(1, out_dim)
        self.optimizer_dict = {}
        self.train_flag = True
        if optimizer_type == 'adam':
            self.optimizer_dict['w'] = AdamOptimizer(self.W, lr)
            self.optimizer_dict['b'] = AdamOptimizer(self.b, lr)
        elif optimizer_type == 'sgd':
            self.optimizer_dict['w'] = SgdOptimizer(lr, 10000)
            self.optimizer_dict['b'] = SgdOptimizer(lr, 10000)

    def forward(self, X):
        """
            X: (batch_size, in_dim)
        """
        Y = np.dot(X, self.W) + self.b
        if self.train_flag:
            self.x = X
        return Y

    def backward(self, grad_y):
        """
            X: (batch_size, in_dim)
            grad_y: (batch_size, out_dim)
        """
        N = self.x.shape[0]
        data_grad = np.dot(grad_y, self.W.T)  # next layer grad
        grad_W = np.dot(self.x.T, grad_y) / N
        grad_b = np.sum(grad_y, axis=0) / N
        self.optimizer_dict['w'].update(self.W, grad_W)
        self.optimizer_dict['b'].update(self.b, grad_b)
        
        return data_grad
    
    def train(self):
        self.train_flag = True
    def eval(self):
        self.train_flag = False
        
        
class ReLU():
    def __init__(self, name = 'relu'):
        print(name+' init')
        self.name = name
    def forward(self, x):
        self.mask = np.float64(x>0)
        return self.mask * x
    def backward(self, grad_y):
        return self.mask * grad_y
    def train(self):
        pass
    def eval(self):
        pass

        
class Sigmoid():
    def __init__(self, name = 'sigmoid'):
        print(name+' init')
        self.name = name
    def forward(self, x):
        self.y = 1 - 1.0/(np.exp(x)+1.0)
        return self.y
    def backward(self, grad_y):
        return (1 - self.y) * self.y * grad_y
    def train(self):
        pass
    def eval(self):
        pass
    
    
def softmax(x):
    max = np.reshape(np.max(x, axis=1), (x.shape[0], 1))
    exp_scores = np.exp(x - max)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs
    
    
class FCNN(object):
    def __init__(self, dim_list, lr):
        self.layers = []
        for i in range(len(dim_list)-1):
            layer = Linear(dim_list[i], dim_list[i+1], name = 'linear'+str(i), lr = lr, optimizer_type='adam')
            self.layers.append(layer)
            if i != len(dim_list)-2:
                layer = ReLU(name = 'relu'+str(i))
#                 layer = Sigmoid(name = 'sigmoid'+str(i))
                self.layers.append(layer)
                
    def forward(self, X):
        Y = X
        for i in range(len(self.layers)):
            Y = self.layers[i].forward(Y)
        return Y
    
    def backward(self, grad_y):
        next_grad_y = grad_y
        for i in range(len(self.layers)-1, -1, -1):
#             print(self.layers[i].name)
            next_grad_y = self.layers[i].backward(next_grad_y)
        return True
    
    def train(self):
        for x in self.layers:
            x.train()
    def eval(self):
        for x in self.layers:
            x.eval()