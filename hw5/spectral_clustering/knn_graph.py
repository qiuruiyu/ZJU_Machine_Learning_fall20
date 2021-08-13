import numpy as np

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    N, P = X.shape
    W = np.zeros((N, N))  # adjacency matrix
    sigma = np.std(X)
    for i in range(N):
        dist_w = np.sum((X[i, :] - X)**2, axis=1)
        idx = np.argsort(dist_w)[:k]
        W[i, idx] = 1
    for i in range(N):
        for j in range(i, N):
            if W[i, j] != W[j, i]:
                W[i, j] = W[j, i] = 0
    return W
    # end answer
