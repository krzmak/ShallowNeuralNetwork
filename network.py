import numpy as np
import random

def intialize(S , K1, K2):
    
    W1 = np.random.uniform(-0.5, 0.5, (K1,S+1))
    W1 = np.round(W1,4)
    W2 = np.random.uniform(-0.5, 0.5, (K2,K1+1))
    W2 = np.round(W2,4)

   # print(W1)
   # print(W2)      

    return W1, W2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def work(W1, W2, X):
    X = np.append(X, -1) 
    U1 = np.dot(W1, X)
    Y1 = sigmoid(U1)
    Y1 = np.append(Y1, -1)  
    U2 = np.dot(W2, Y1)
    Y2 = sigmoid(U2)
    return Y1, Y2

def learn(W1first, W2first, P, T, n):
    NumberOfExamples = P.shape[1]
    W1 = W1first
    W2 = W2first
    LearnFactor = 0.6
    Beta = 5

    for i in range(n):
        ExampleNr = random.randint(0, NumberOfExamples - 1)
        X = P[:, ExampleNr]
        T_example = T[:, ExampleNr]
        Y1, Y2 = work(W1, W2, X)

        D2 = Beta * (Y2 * (1 - Y2)) * (T_example - Y2)
        dW2 = LearnFactor * np.outer(D2, Y1)

        D1 = Beta * (Y1[:-1] * (1 - Y1[:-1])) * np.dot(W2[:, :-1].T, D2)
        dW1 = LearnFactor * np.outer(D1, np.append(X, -1))

        W1 += dW1
        W2 += dW2



        """
        print(f"Iteration {i+1}/{n}")
        print("Example number:", ExampleNr)
        print("X:", X)
        print("T_example:", T_example)
        print("Y1:", Y1)
        print("Y2:", Y2)
        print("D2:", D2)
        print("E2:", E2)
        print("D1:", D1)
        print("E1:", E1)
        print("Updated W1:", W1)
        print("Updated W2:", W2)
        """
    return W1 , W2    
    
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)


    W1, W2 = intialize(2, 2, 1)
    
    P = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    T = np.array([[0, 1, 1, 0]])                   
    n = 5000
    W1, W2 = learn(W1, W2, P, T, n)
   # print("Final W1:", W1)
   # print("Final W2:", W2)

    for i in range(P.shape[1]):
        X = P[:, i]
        _, Y2 = work(W1, W2, X)
        print(f"Input: {X}, Predicted Output: {Y2}, Target: {T[:, i]}")