import numpy as np

class Layer:
    def __init__(self,i,o):
        self.n = None
        self.y = None
        self.g = None
        self.w = np.random.randn(i,o)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def sigmoidPrime(x):
    s = sigmoid(x);
    return s*(s-1.0)
def process(x,y):
    T = [2,3,4,1]
    L = len(T)
    l = []

    for i in range(L):
        try:
            l.append(Layer(T[i-1],T[i]))
        except IndexError:
            l.append(Layer(0,T[i]))
    
    l[0].y = x
    for i in range(1,L):
        l[i].n = np.dot(l[i-1].y,l[i].w)
        l[i].y = sigmoid(l[i].n)
    l[-1].g = y - l[-1].y # g = gradient
    for i in reversed(range(1,L-1)):
        l[i].g = np.dot(l[i+1].g,l[i+1].w.T) * sigmoidPrime(l[i].n)
    for i in range(1,L):
        l[i].w += np.dot(l[i].g,l[i-1].y.T)

def main():
    x = np.asarray([[1,2]])
    y = np.asarray([[3]])
    process(x,y)


if __name__ == "__main__":
    main()
