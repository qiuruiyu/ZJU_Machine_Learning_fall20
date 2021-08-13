import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape  # N for number of samples, P for number of features 
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE
    # begin answer
    lr = 0.001  # learning rate 

    X = np.vstack((np.ones((1, N)), X))

    error_num = N
    while error_num > N / 20:
        iters += 1
        error_num = 0
        for col in range(X.shape[1]):
            pred = w.T.dot(X[:, col])
            if ((pred > 0).astype(int) * 2 - 1) != y[0, col]:  # calssification not correspond
                w += (lr * y[0, col] * X[:, col]).reshape(-1, 1)
                error_num += 1
        if iters > 1000:
            break

    return w, iters