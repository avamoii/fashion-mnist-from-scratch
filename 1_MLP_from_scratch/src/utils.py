import numpy as np
def one_hot_encode(labels, num_classes=10):
    m= labels.shape[0]
    one_hot = np.zeros((num_classes, m))
    return one_hot

def load_data(train_path, test_path):
  
    train_data = np.loadtxt(train_path, delimiter=",", skiprows=1)
    test_data = np.loadtxt(test_path, delimiter=",", skiprows=1)

    Y_train = train_data[:, 0].astype(int)
    Y_test = test_data[:, 0].astype(int)

    X_train = train_data[:, 1:].T / 255.0
    X_test = test_data[:, 1:].T / 255.0

    Y_train = one_hot_encode(Y_train, num_classes=10)
    Y_test = one_hot_encode(Y_test, num_classes=10)

    return X_train, Y_train, X_test, Y_test