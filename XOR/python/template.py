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


#Network
class Net:
    def __init__(self):
        self.W = [] #weight matrices
        self.L = [] #Layers
        pass
    def FF(self,X):
        pass
    def BP(self,X,Y):
        G = 0.5 * (Y - self.FF(X))**2 #ERROR
        pass


def main():
    pass
if __name__ == "__main__":
    main()
