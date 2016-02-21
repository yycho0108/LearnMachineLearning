import numpy as np

def rVec(n):
    return np.ndarray((1,n))
def rVec_l(*args):
    return np.asarray([args])

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
def sigmoidPrime(x):
    s = sigmoid(x)
    return s*(s-1.0) #verify this.

class Layer:
    def __init__(self,n):
        self.Z = rVec(n) # I = input
        self.A = rVec(n) # A = activation
    def transfer(self,i):
        self.Z = i
        self.A = sigmoid(i)
        return self.A

class Net:
    def __init__(self,t):
        self.L = [] #L[0] = Input.
        self.W = []
        self.length = len(t)
        for i in range(self.length-1):
            self.W.append(np.random.randn(t[i],t[i+1]))
            self.L.append(Layer(t[i]))
        self.L.append(Layer(t[-1]))
        print(self.L)
        #this is essentially the "OUTPUT LAYER".
    def FF(self,x):
        self.L[0].transfer(x) #input layer! transfer necessary?? well..
        for l, w in zip(self.L[1:],self.W):
            x = l.transfer(np.dot(x,w))
        return x

    def BP(self,x,y):
        Y = self.FF(x)
        #G = (y-Y)*sigmoidPrime(self.L[-1].Z) #gradient
        for i in reversed(range(1,self.length)):
            print("I",i)
            if i == self.length-1: #output
                G = (y-Y)*sigmoidPrime(self.L[i].Z) #gradient
            else:#hidden
                print("W", self.W[i-1])
                #print("?", np.dot(self.W[i-1],G))
                print("Z", self.L[i].Z)
                G = np.dot(G,self.W[i].T)
                G *= sigmoidPrime(self.L[i-1].Z)
            print("G",G)
            dW = np.dot(G.T,self.L[i-1].A)
            print("dW",dW)

def XOR_GEN():
    a,b = ab_rand()
    a = np.round(a)
    b = np.round(b)
    I = rVec_l(a,b)
    O = rVec_l(int(a)^int(b))
    return I,O

def MULT_GEN():
    a,b = ab_rand()
    I = rVec_l(a,b)
    O = rVec_l(a*b)
    return I,O

def AVG_GEN():
    a,b = ab_rand()
    I = rVec_l(a,b)
    O = rVec_l((a+b)/2)
    return I,O

def GEN():
    return XOR_GEN()

def main():
    topology = [2,3,4]
    net = Net(topology)
    x = rVec_l(1,2)
    y = rVec_l(3,4,5,6)
    
    print("FF")
    print(net.FF(x))
    print("BP")
    net.BP(x,y)


if __name__ == "__main__":
    main()
