import numpy as np
import sys

class SgdOptimizer():
    def __init__(self, lr, grad_clip_thres):
        self.name = 'sgd'
        self.lr = lr
        self.grad_clip_thres = grad_clip_thres
        
    def clip_grad(self, x, thres = 10):
        length = np.sqrt(np.sum(x*x))
        if length >= thres:
            return x/length*thres*0.1
        else:
            return x
        
    def update(self, x, grad_x):
        grad_x = self.clip_grad(grad_x, self.grad_clip_thres)
        x -= self.lr * grad_x
        
        
class AdamOptimizer():
    def __init__(self, var, lr):
        self.name = 'adam'
        self.adam_belta1=0.9
        self.adam_belta2=0.999
        self.adam_epsilon=1e-8
        self.adam_alpha=lr
        self.vt = np.zeros(var.shape)
        self.mt = np.zeros(var.shape)
        
        
    def update(self, x, grad_x):
        self.mt=self.adam_belta1 * self.mt+(1.-self.adam_belta1)*grad_x
        self.vt=self.adam_belta2 * self.vt+(1.-self.adam_belta2)*(np.square(grad_x))
        x -= self.adam_alpha * self.mt / (np.sqrt(self.vt) + self.adam_epsilon)
        