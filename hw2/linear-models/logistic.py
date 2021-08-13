import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
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
    while error_num > N / 20:
        iters += 1
        error_num = 0
        for col in range(N):
            pred = w.T.dot(X[:, col])
            pred = 1 / (1 + np.exp(-pred))
            if (pred > 0.5).astype(int) != y[0, col]:  # classification not correspond
                w += (lr * (y[0, col] - pred) * X[:, col]).reshape(-1, 1)
                error_num += 1

        if iters > 50:
            break
    # end answer
    
    return w
