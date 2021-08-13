import numpy as np
from kmeans import kmeans


def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    D = np.diag(np.sum(W, axis=0))
    L = D - W  # Laplacian matrix 
    w, v = np.linalg.eigh(L) # clac the eigvalue and eigvector
    idx = kmeans(v[:, :k], k)
    return idx
    # end answer

