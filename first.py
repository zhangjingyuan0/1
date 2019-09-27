import math
import numpy as np

#Network0:单个神经元
#Network1:一层神经网络一个输出
#Network2:一层神经网络任意个输出
#Network3:多层网络

def sigmoid(z):
    return 1/(1 + pow(math.e,-z))

class Network0():
    def __init__(self):
        self.w = 1
        self.b = 2
    def y(self,x):
        return sigmoid(self.w * x + self.b)

class Network1():
    def __init__(self,size):
        self.size = size
        self.w = np.random.randn(size,1);
        self.b = np.random.randn()
    def __call__(self,x):
        return sigmoid(self.w * x + self.b)

class Network2():
    def __init__(self,m,n):
        self.n = n
        self.m = m
        self.w = np.random.randn(n,m)
        self.b = np.random.randn(n,1)

    def __call__(self,x):
        return sigmoid(np.matmul(self.w,x) + self.b)

class Network3():
    def __init__(self,sizes):
        self.sizes = sizes
        self.n = len(sizes)
        self.b_list = []
        self.w_list = []
        for i in range(self.n):
            self.w_list.append(np.random.randn(sizes[i][1],sizes[i][0]))
            self.b_list.append(np.random.randn(sizes[i][1],1))

    def __call__(self,x):
        for i in range(self.n):
            x = sigmoid(np.matmul(self.w_list[i], x) + self.b_list[i])
        return x
            
    

ner = Network3([[5,3],[3,1]])
y = ner([[1],[2],[3],[4],[5]])
print(y)