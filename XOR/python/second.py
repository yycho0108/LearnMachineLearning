import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def sigmoidPrime(x):
    s = sigmoid(x)
    return s*(1-s)

class Layer:
    def __init__(self,n):
        self.I = np.ndarray((1,n))
        self.O = np.ndarray((1,n))
    def transfer(self,i):
        self.I = i;
        self.O = sigmoid(i)
        return self.O
    def back(self,Y):
        pass

class InputLayer(Layer):
    def __init__(self,n):
        Layer.__init__(self,n)

    def transfer(self,i):
        self.I = i;
        self.O = i;
        return self.O

class OutputLayer(Layer):
    def __init__(self,n):
        Layer.__init__(self,n)

    def transfer(self,i):
        self.I = i;
        self.O = i;
        return self.O

class Net:
    def __init__(self,t): # t = topology, e.g. 2-3-1
        self.W = []
        self.L = []

        #initialize Layers
        for i in range(len(t)):
            self.L.append(Layer(t[i]))
        self.L[0] = InputLayer(t[0])
        self.L[-1] = OutputLayer(t[-1])

        #initialize Weights
        for i in range(len(t)-1):
            self.W.append(np.random.randn(t[i],t[i+1])) #from this layer to next.
    
    def forward(self,x):
        self.L[0].transfer(x)
        for i in range(len(self.W)):
            print(x)
            print(self.W[i])
            x = np.dot(x,self.W[i])
            x = self.L[i+1].transfer(x) 
        return x

    def back(self,y,t):#y=res, t=target
        e = 0.5*(t-y)**2 #error
        print(y)
        print(t)
        print(e)
        for i in reversed(range(len(self.W))):
            w = self.W[i]
            l = self.L[i+1]
            e = self.delta(l,w,e)
            print(e)
            dw = np.dot(e,self.L[i-1].O)
            print(dw)
            self.W[i] += dw

    def delta(self,L,W,E):#layer, weight, error
        # represents the error in layer.
        print(E)
        print(np.dot(np.transpose(W),E))
        return np.dot(np.transpose(W),E)*sigmoidPrime(L.I)

if __name__ == "__main__":
    topology = [1,2,3,4,5]
    net = Net(topology)
    i = 2.0
    o = net.forward(i)
    net.back(o,o)
    #o = 1.0
    #net.back(i,o)
    #print(o)
