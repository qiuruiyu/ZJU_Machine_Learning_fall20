import numpy as np 
from hack import hack
from extract_image import extract_image
from show_image import show_image
import os

def get_label(path):
    train_data = os.listdir(path)  # get all data in the path

    x_train = np.zeros((0, 144))
    with open('./_digits.txt', 'r') as f:

        y_train = []
        for l in f.readlines():
            l = l.strip('\n')
            y_train.append(l)
        y_train = np.array(y_train)

    for data in train_data:
        x = extract_image(path+'/'+data)
        x_train = np.vstack((x_train, x))
        show_image(x)  # show the image in .ipynb 

    return x_train, y_train
