import numpy as np
from scipy.stats import mode


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful
    # Hint: distance = (x - x_train)^2. How to vectorize this?

    # YOUR CODE HERE

    # begin answer
    N_test, _ = x.shape
    y = np.zeros((N_test, 1))

    for i in range(N_test):
        xi = x[i, :]
        dist = np.sum((xi - x_train)**2, axis=1)
        dist_idx = np.argsort(dist)[:k]  # return the k idx of distance from low to high 
        yi = y_train[dist_idx]
        cls, times = mode(yi)[0][0], mode(yi)[1][0]
        y[i] = cls
    y = y.squeeze()
    # end answer

    return y
