import numpy as np
from matplotlib import pyplot as plot


visit = 0

def ALPHA():
    return 0.6
    #global visit
    #visit += 1
    #return np.sqrt(1.0 / visit)

def sigmoid(x):
    #return 3.0/(1.0 + np.exp(-x))
    return 1.0/(1.0 + np.exp(-x))

def sigmoidPrime(x):
    #return 3*np.exp(-x) / (np.exp(-x)+1)**2
    s = sigmoid(x)
    return s*(1.0-s)

def rVec(n):
    return np.ndarray((1,n))

def rVec_l(*args):
    return np.asarray([args]);

class Layer:
    def __init__(self,n):
        self.I = rVec(n) # row vector 
        self.O = rVec(n) 
    def transfer(self,I):
        self.I = I
        self.O = sigmoid(self.I)
        return self.O

class Net:
    def __init__(self,T):
        self.length = len(T)-1
        self.L = []
        self.W = []
        for i in range(self.length):
            self.W.append(np.clip(np.random.randn(T[i],T[i+1]),0.0,1.0))
            self.L.append(Layer(T[i]))

    def FF(self,X):
        X = np.array(X)
        self.L[0].I = self.L[0].O = X #input layer
        for i in range(self.length):
            try:
                X = self.L[i+1].transfer(np.dot(X,self.W[i]))
            except:
                X = np.dot(X,self.W[i])
            #if i != 0:
            #    X = np.dot(self.L[i].transfer(X), self.W[i]) # row vector  * W[i][o]
            #else:
            #    self.L[i].I = self.L[i].O = X
            #    X = np.dot(X, self.W[i])
        return X 

    def BP(self,O,T): #output, target
        dEO = T-sigmoid(O)
        dON = sigmoidPrime(O)
        G = dEO * dON
        print(G)
        for i in reversed(range(self.length)): 
            dw = np.dot(G.T,self.L[i].O) # minus, to minimize error
            dEO = np.dot(self.W[i],G.T) #working here for previous layer
            dON = sigmoidPrime(self.L[i-1].O)
            G = dEO * dON 
            self.W[i] += ALPHA() * dw.T #update after being used!

def ab_rand():
    a = np.random.random()
    b = np.random.random()
    return a,b

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
    Topology = [2,4,1]
    net = Net(Topology)
    
    #I = rVec_l(0.5)
    #Target = rVec_l(0.1,0.2,0.3,0.4)
    #for i in range(300):
    #    O = net.FF(I)
    #    net.BP(O,Target)
    
    #v = []
    #for i in range(3000):
    #    I,Target = MULT_GEN()
    #    O = net.FF(I)
    #    net.BP(O,Target)
    #    v.append(np.sum(O-Target))
    #I,Target = MULT_GEN()
    #print(I, net.FF(I), Target)
    #plot.plot(v)
    #plot.show()
    v_1 = []
    v_2 = []
    for i in range(1000):
        I,T = GEN()
        O1 = net.FF(I)
        net.BP(O1,T)
        O2 = net.FF(I)
        E1 = np.sum(np.abs(O1-T))
        E2 = np.sum(np.abs(O2-T))
        v_1.append(E1)
        v_2.append(E2)
    I,T = GEN()
    O = net.FF(I)
    print(O,T)
    plot.plot(v_1,label="V_1")
    plot.plot(v_2,label="V_2")
    plot.show()
    

if __name__ == "__main__":
    main()
