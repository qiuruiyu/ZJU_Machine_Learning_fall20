import numpy as np
from scipy.optimize import minimize


def svm(X, y, noisy=False):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    num = 0

    # YOUR CODE HERE
    X = np.vstack((np.ones((1, N)), X))

    if noisy is not True:
        func = lambda w,X,y: 0.5 * np.sum(w**2)  # goal of minimizing function
        res = minimize(func, w, args=(X,y), constraints=({'type':'ineq', 'args':(X,y), 'fun':lambda w,X,y: y*(w.T.dot(X))-1}))
        w = res.x
        num = (((y * (w.T.dot(X))) < 1.0001) * (y * (w.T.dot(X))) > 0).astype(int).sum()

    else:  # if noisy is used 
        C = 1  # criterion factor 
        w_relax = np.vstack((w, np.zeros((N, 1))))
        func = lambda w,P,C,X,y: 0.5 * np.sum(w[:P+1]**2) + C * w[P+1:].sum()
        cons = (
            {'type':'ineq', 'args':(P,X,y), 'fun': lambda w,P,X,y: y*(w[:P+1].T.dot(X))-1+w[P+1:]},
            {'type':'ineq', 'args':(P,), 'fun': lambda w,P: w[P+1:]}
        )
        res = minimize(func, w_relax, args=(P,C,X,y), constraints=cons)
        w = (res.x)[:P+1]
        e = (res.x)[P+1:]  # ksei vector
        num = (((y * (w.T.dot(X))) < 1.0001-e) * (y * (w.T.dot(X))) > 0).astype(int).sum()

    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    # end answer
    return w, num

