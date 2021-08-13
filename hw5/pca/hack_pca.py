import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from pca import PCA
import cv2

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)
    # YOUR CODE HERE
    # begin answer
    gray_weight = np.array([0.3, 0.6, 0.1, 0])
    threshold = 130
    print(img_r[0][0].shape)
    img_binary = np.zeros((img_r.shape[0], img_r.shape[1]))
    img_binary = np.sum(np.multiply(img_r[:, :, :], gray_weight), axis=2)
    
    fea = np.array(np.where(img_binary > threshold))
    vec, val = PCA(fea.T)
    new_xy = np.zeros((img_binary.shape[0], img_binary.shape[1], 2), dtype=int)
    x = np.arange(img_binary.shape[0])
    y = np.arange(img_binary.shape[1])
    for i in x:
        for j in y:
            new_xy[i, j] = vec.T.dot(np.array([i, j]))
    _x, _y = np.mean(new_xy[:, :, 0]), np.mean(new_xy[:, :, 1])
    new_xy = new_xy - np.array([_x, _y], dtype=int) + np.array([int(img_binary.shape[0]/2), int(img_binary.shape[1]/2)])
    res = np.zeros(img_binary.shape)
    for i in x:
        for j in y:
            if 0 <= new_xy[i, j, 0] < img_binary.shape[0] and 0 <= new_xy[i, j, 1] < img_binary.shape[1]:
                res[new_xy[i, j, 0], new_xy[i, j, 1]] = img_binary[i, j]

    return res

    # end answer

