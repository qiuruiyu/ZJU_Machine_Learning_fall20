import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    p = np.zeros((C, N))
    #TODO

    # begin answer
    p[0] = l[0]
    p[1] = l[1]
    # calculate the p_ω1 and p_ω2
    p_w1 = np.sum(x[0]) / total
    p_w2 = np.sum(x[1]) / total
    p_x = [p_w1 * l[0][i] + p_w2 * l[1][i] for i in range(x.shape[1])]
    for i in range(len(p[0])):
        p[0][i] = p[0][i] * p_w1 / p_x[i]
    for i in range(len(p[1])):
        p[1][i] = p[1][i] * p_w2 / p_x[i]
    # end answer
    
    return p
