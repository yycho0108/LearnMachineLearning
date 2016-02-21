import numpy as np

#Function generators
def XOR_GEN():
    I = np.random.rand(1,2)
    I = np.round(I).astype(int)
    O = I[0][0]^I[0][1]
    O = np.asarray(O)
    return I,O

def GEN():
    return XOR_GEN()

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def sigmoidPrime(x):
    s = sigmoid(x)
    return s*(s-1.0)
def rVec(n):
    return np.ndarray((1,n))
def rVec_l(*args):
    return np.asarray([args]);

class Layer:
    def __init__(self,i,o):
        self.W = np.random.randn(i,o) #weight array
        self.I = rVec(i)
        self.SI = rVec(i) # sigmoid(self.I)
        self.O = rVec(o)
    def forward(self,X):
        self.I = X
        self.SI = sigmoid(X)
        #print("X", X)
        #print("SI", self.SI)
        self.O = np.dot(self.SI,self.W) # (+ bias)
        return self.O
    def calculateGradient(self,E): # E = "ERROR" of next
        """ CALCULATE GRADIENT """
        print("E", E)
        print("W", self.W)
        self.dE = np.dot(E,self.W.T) * sigmoidPrime(self.I)
        #dW = np.dot(dE, self.O)
        #self.W += dW
        return self.dE #for next.
    def update(self,O): # O = Output of PREVIOUS layer
        print("I",self.I)
        print("dE", self.dE)
        dW = np.dot(self.dE,O)
        self.W += dW
        return self.O
#Network
class Net:
    def __init__(self,L):
        self.L = [] #Layers
        for l_1, l_2 in zip(L[:-1],L[1:]):
            self.L.append(Layer(l_1,l_2))

    def feedForward(self,X):
        for l in self.L:
            X = l.forward(X)
        return sigmoid(X)
        #return X
    def backPropagate(self,X,Y):
        E = Y - self.feedForward(X) #ERROR
        for l in reversed(self.L):
            E = l.calculateGradient(E)

        O = X # y_o
        for l in self.L:
            O = l.update(O)


def main():
    L = [2,3,3,1]
    net = Net(L)
    I,O = GEN()
    print(net.feedForward(I))
    net.backPropagate(I,O)
if __name__ == "__main__":
    main()
