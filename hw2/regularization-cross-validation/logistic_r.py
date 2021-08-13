import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    lr = 0.01
    X = np.vstack((np.ones((1, N)), X))
    y = (y + 1) / 2 
    iters = 0
    error_num = N
    while error_num > N / 50:  # loop end condition 
        iters += 1 
        error_num = 0
        for col in range(N):  # for every X 
            pred = w.T.dot(X[:,col])
            pred = 1 / (1 + np.exp(-pred))  # change to possibility 
            if ((pred > 0.5).astype(int) * 2 - 1) != y[0, col]:
                w += (lr * (y[0, col] - pred) * X[:, col]).reshape(-1, 1) + lr * lmbda * w
                error_num += 1
        if iters > 50:  # set a threshold 
            break

    # end answer
    return w
