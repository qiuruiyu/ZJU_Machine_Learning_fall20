import numpy as np


# def kmeans(x, k):
#     '''
#     KMEANS K-Means clustering algorithm

#         Input:  x - data point features, n-by-p maxtirx.
#                 k - the number of clusters

#         OUTPUT: idx  - cluster label
#                 ctrs - cluster centers, K-by-p matrix.
#                 iter_ctrs - cluster centers of each iteration, (iter, k, p)
#                         3D matrix.
#     '''
#     # YOUR CODE HERE

#     # begin answer
#     max_iters = 1000
#     N, P = x.shape
#     idx = np.array(np.zeros(N), dtype=int)  # label index 
#     ctrs = x[np.random.choice(N, k), :]
#     iters = 0
#     # iter_ctrs = np.zeros((max_iters+1, k, P))
#     iter_ctrs = np.zeros((1, k, P))
#     iter_ctrs[0, :, :] = ctrs

#     while iters <= max_iters:
#         iters += 1
#         error = 0
#         for i in range(N):
#             dist = [np.sum((x[i, :] - ctrs)**2, axis=1)]  # calc the dist from one point to each clustering center
#             if idx[i] != np.argmin(dist):
#                 error += 1
#                 idx[i] = np.argmin(dist)

#         if error == 0:
#             break
        
#         for c in range(k):
#             ctrs[c] = np.average(x[idx==c, :], axis=0)

#         iter_ctrs = np.concatenate((iter_ctrs, np.expand_dims(ctrs, axis=0)), axis=0)
#     # end answer

#     return idx, ctrs, iter_ctrs


def kmeans(x, k):
    '''
    A Fast implementation of K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
    '''

    x = x.astype(float)
    n = x.shape[0]
    ctrs = x[np.random.permutation(x.shape[0])[:k]]
    iter_ctrs = [ctrs]
    idx = np.ones(n)
    iters = 0
    x_square = np.expand_dims(np.sum(np.multiply(x, x), axis=1), 1)

    while iters < 100:
        iters += 1
        distance = -2 * np.matmul(x, ctrs.T)
        distance += x_square
        distance += np.expand_dims(np.sum(ctrs * ctrs, axis=1), 0)
        new_idx = distance.argmin(axis=1)
        if (new_idx == idx).all():
            break
        idx = new_idx
        ctrs = np.zeros(ctrs.shape)
        for i in range(k):
            ctrs[i] = np.average(x[idx == i], axis=0)
        iter_ctrs.append(ctrs)
    iter_ctrs = np.array(iter_ctrs)

    return idx, ctrs, iter_ctrs

