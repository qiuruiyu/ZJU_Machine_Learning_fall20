import numpy as np

def likelihood(x):
    '''
    LIKELIHOOD Different Class Feature Likelihood 
    INPUT:  x, features of different class, C-By-N numpy array
            C is the number of classes, N is the number of different feature

    OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N numpy array
    '''

    C, N = x.shape
    l = np.zeros((C, N))
    #TODO

    # begin answer
    l[0] = x[0]
    l[1] = x[1]
    for cls in l:  # for every class in given x 
        total_num = 0  
        # claculate the total number of it
        for ele in cls:
            total_num += ele
        cls /= total_num  # normalize
    # end answer

    return l