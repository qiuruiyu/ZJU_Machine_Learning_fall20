import numpy as np

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    import time 

    start = time.time()
    
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    value_sigma_all = []
    sigma_inv_all = []
    for cls in range(K):
        sigma_cls = Sigma[:, :, cls]
        value_sigma_all.append(np.linalg.det(sigma_cls))
        sigma_inv_all.append(np.linalg.inv(sigma_cls))

    for col in range(N):  # every col stands for a data of M features, col from 0 to N-1
        x = X[:, col].reshape(-1, 1)  # take the col^th data out, M-by-1
        for cls in range(K):  # K classification, cls from 0 to K-1
            mu_cls = Mu[:, cls].reshape(-1, 1)  # take the cls^th Mu out, M-by-1
            sigma_cls = Sigma[:, :, cls]  # take the cls^th Mu out, M-by-M
            likelihood_cls = np.exp(-0.5 * (x-mu_cls).T.dot(sigma_inv_all[cls]).dot(x-mu_cls)) / (2 * np.pi * np.sqrt(value_sigma_all[cls])) 
            prior_cls = Phi[cls]  # take the prior possibility of every cls 
            posterior_cls = likelihood_cls * prior_cls 
            p[col, cls] = posterior_cls  

    for row in p:
        row /= np.sum(row)  # normalization 
        
    end = time.time()
    print('Time Cost: {:.3f}s'.format(end-start))
    # end answer 
   
    return p

