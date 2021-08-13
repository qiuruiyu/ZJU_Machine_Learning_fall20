import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis
    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    data = data - np.mean(data, axis=0)
    val, vec = np.linalg.eigh(np.cov(data.T))  # eigh of cov(XX^T)
    idx = np.argsort(-val)
    return vec[:, idx], val[idx]
    # end answer