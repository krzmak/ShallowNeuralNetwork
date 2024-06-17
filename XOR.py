import numpy as np
import random
import matplotlib.pyplot as plt

def initialize(S, K1, K2):
    W1 = np.random.uniform(-0.5, 0.5, (K1, S + 1))
    W1 = np.round(W1, 4)
    W2 = np.random.uniform(-0.5, 0.5, (K2, K1 + 1))
    W2 = np.round(W2, 4)
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

def mean_squared_error(T, Y):
    return np.mean((T - Y) ** 2)

def classification_error(T, Y):
    return np.mean(np.round(T) != np.round(Y))

def learn(W1first, W2first, P, T, n):
    NumberOfExamples = P.shape[1]
    W1 = W1first
    W2 = W2first
    LearnFactor = 0.3
    Beta = 5

    mse1_history = []
    mse2_history = []
    ce_history = []

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

        # Calculate and store errors for plotting
        Y1_all = np.array([work(W1, W2, P[:, j])[0][:-1] for j in range(NumberOfExamples)])
        Y2_all = np.array([work(W1, W2, P[:, j])[1] for j in range(NumberOfExamples)])

        mse1 = mean_squared_error(T, Y1_all.T)
        mse2 = mean_squared_error(T, Y2_all.T)
        ce = classification_error(T, Y2_all.T)

        mse1_history.append(mse1)
        mse2_history.append(mse2)
        ce_history.append(ce)

    return W1, W2, mse1_history, mse2_history, ce_history

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    W1, W2 = initialize(2, 2, 1)
    
    P = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    T = np.array([[0, 1, 1, 0]])                   
    n = 5000
    W1, W2, mse1_history, mse2_history, ce_history = learn(W1, W2, P, T, n)

    for i in range(P.shape[1]):
        X = P[:, i]
        _, Y2 = work(W1, W2, X)
        print(f"Input: {X}, Predicted Output: {Y2}, Target: {T[:, i]}")

    # Plotting errors
    plt.figure(figsize=(10, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(mse1_history, label="MSE Layer 1")
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error Layer 1')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(mse2_history, label="MSE Layer 2")
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error Layer 2')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(ce_history, label="Classification Error")
    plt.xlabel('Epochs')
    plt.ylabel('CE')
    plt.title('Classification Error')
    plt.legend()
    
    plt.tight_layout(pad=2.0)
    plt.show()
