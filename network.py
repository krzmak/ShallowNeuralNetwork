import numpy as np
import random

def intialize(S , K1, K2):
    
    W1 = np.random.uniform(-0.1, 0.1, (K1,S+1))
    W1 = np.round(W1,4)
    W2 = np.random.uniform(-0.1, 0.1, (K2,K1+1))
    W2 = np.round(W2,4)

    print(W1)
    print(W2)      

    return W1, W2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def work(W1 , W2, X):
    X = np.append(X,-1)
    U1 = np.dot(X, W1.T)
    Y1 = sigmoid(U1)
    Y1 = np.insert(U1, 0, -1)
    print("Y1:", Y1)
    U2 = np.dot(Y1, W2.T)
    Y2 = sigmoid(U2)
    print("Y2:", Y2)


    return Y1, Y2

if __name__ == "__main__":
    
    W1 , W2 =  intialize(2,5,2)
    X = [2, 3]
    work(W1, W2, X)
