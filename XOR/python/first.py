import numpy as np
import sys

class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize);
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize);

    def forward(self,X):
        self.z2 = np.dot(X,self.W1);
        self.a2 = self.sigmoid(self.z2);
        self.z3 = np.dot(self.a2,self.W2);
        yHat = self.sigmoid(self.z3);
        return yHat;
    def sigmoid(self,z):
        return 1/(1+np.exp(-z));
    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2);
    def costFuncPrime(self,X,y):
        self.yHat = self.forward(X);
        d3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.transpose(),d3)
        d2 = np.dot(d3,self.W2.transpose())*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.transpose(),d2)
        return dJdW1, dJdW2
    def costFunc(self,x,y):
        self.yHat = self.forward(x);
        return 0.5 * sum((y-self.yHat)**2)

if __name__ == "__main__":
    x = np.array(([0.3,1.],[0.5,0.2],[1.,0.4]))
    y = np.array(([0.75],[0.82],[0.93]))
    NN = Neural_Network()
    n = int(sys.argv[1])
    while n > 0:
        print(NN.costFunc(x,y))
        djdw1,djdw2 = NN.costFuncPrime(x,y)
        NN.W1 = NN.W1-djdw1
        NN.W2 = NN.W2-djdw2
        n = n-1
    print(NN.costFunc(x,y))
    print(NN.W1)
    print(NN.W2)
