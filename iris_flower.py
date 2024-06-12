import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.float_format = '{:.10f}'.format

def separate_data_custom(csv_file):
    df = pd.read_csv(csv_file)
    
    # Extract features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
    y = df['species'].astype('category').cat.codes.to_numpy()
    
    return X, y

def split_data_custom(X, y):
    # Convert to dataframe for easier manipulation
    data = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    data['species'] = y
    
    # Initialize lists for training and test data
    train_data = []
    test_data = []
    
    # Get the unique species
    species = data['species'].unique()
    
    for sp in species:
        species_data = data[data['species'] == sp]
        train_data.append(species_data.iloc[:-5])
        test_data.append(species_data.iloc[-5:])
    
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    
    X_train = train_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
    y_train = train_data['species'].to_numpy()
    X_test = test_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
    y_test = test_data['species'].to_numpy()
    
    return X_train, X_test, y_train, y_test

def separate_data_into_arrays(X, y):
    # Separate features
    sepal_length = X[:, 0]
    sepal_width = X[:, 1]
    petal_length = X[:, 2]
    petal_width = X[:, 3]
    
    return sepal_length, sepal_width, petal_length, petal_width, y

def initialize(S, K1, K2):
    W1 = np.random.uniform(-0.5, 0.5, (K1, S + 1))
    W1 = np.round(W1, 4)
    W2 = np.random.uniform(-0.5, 0.5, (K2, K1 + 1))
    W2 = np.round(W2, 4)
    return W1, W2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def work(W1, W2, X):
    X = np.append(X, -1)
    U1 = np.dot(W1, X)
    Y1 = sigmoid(U1)
    Y1 = np.append(Y1, -1)
    U2 = np.dot(W2, Y1)
    Y2 = softmax(U2)
    return Y1, Y2

def mean_squared_error(T, Y):
    return np.mean((T - Y) ** 2)

def classification_error(T, Y):
    return np.mean(np.argmax(T, axis=0) != np.argmax(Y, axis=0))

def learn(W1first, W2first, P, T, n, learn_factor, beta):
    NumberOfExamples = P.shape[1]
    W1 = W1first
    W2 = W2first

    mse2_history = []
    ce_history = []

    for i in range(n):
        ExampleNr = random.randint(0, NumberOfExamples - 1)
        X = P[:, ExampleNr]
        T_example = T[:, ExampleNr]
        Y1, Y2 = work(W1, W2, X)

        D2 = beta * (Y2 * (1 - Y2)) * (T_example - Y2)
        dW2 = learn_factor * np.outer(D2, Y1)

        D1 = beta * (Y1[:-1] * (1 - Y1[:-1])) * np.dot(W2[:, :-1].T, D2)
        dW1 = learn_factor * np.outer(D1, np.append(X, -1))

        W1 += dW1
        W2 += dW2

        # Calculate and store errors for plotting
        Y2_all = np.array([work(W1, W2, P[:, j])[1] for j in range(NumberOfExamples)])
        Y2_all = Y2_all.T

        mse2 = mean_squared_error(T, Y2_all)
        ce = classification_error(T, Y2_all)

        mse2_history.append(mse2)
        ce_history.append(ce)

    return W1, W2, mse2_history, ce_history

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    csv_file = 'IRIS.csv'
    X, y = separate_data_custom(csv_file)
    X_train, X_test, y_train, y_test = split_data_custom(X, y)
    
    # Normalize the features
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    # One-hot encode the target values
    y_train_one_hot = np.eye(3)[y_train].T
    y_test_one_hot = np.eye(3)[y_test].T

    # Separate training data into arrays
    sepal_length_train, sepal_width_train, petal_length_train, petal_width_train, y_train = separate_data_into_arrays(X_train, y_train)

    # Separate test data into arrays
    sepal_length_test, sepal_width_test, petal_length_test, petal_width_test, y_test = separate_data_into_arrays(X_test, y_test)

    W1, W2 = initialize(4, 15, 3)
    
    P = np.vstack((sepal_length_train, sepal_width_train, petal_length_train, petal_width_train))
    T = y_train_one_hot
    n = 5000
    W1, W2, mse2_history, ce_history = learn(W1, W2, P, T, n, 0.09, 1)

    print("Training results:")
    for i in range(P.shape[1]):
        X = P[:, i]
        _, Y2 = work(W1, W2, X)
        print(f"Input: {X}, Predicted Output: [{', '.join(f'{value:.10f}' for value in Y2)}], Target: {T[:, i]}")

    # Testing the trained model on test data
    P_test = np.vstack((sepal_length_test, sepal_width_test, petal_length_test, petal_width_test))
    T_test = y_test_one_hot

    Y2_test_all = np.array([work(W1, W2, P_test[:, j])[1] for j in range(P_test.shape[1])])
    Y2_test_all = Y2_test_all.T

    # Plotting errors
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(mse2_history, label="MSE Layer 2")
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error Layer 2')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(ce_history, label="Classification Error")
    plt.xlabel('Epochs')
    plt.ylabel('CE')
    plt.title('Classification Error')
    plt.legend()
    
    plt.tight_layout(pad=2.0)
    plt.show()

    print("\nTesting results:")
    for i in range(P_test.shape[1]):
        X = P_test[:, i]
        _, Y2 = work(W1, W2, X)
        print(f"Input: {X}, Predicted Output: [{', '.join(f'{value:.10f}' for value in Y2)}], Target: {T_test[:, i]}")
