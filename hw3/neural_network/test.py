import time
import numpy as np
import matplotlib.pyplot as plt
from fc_net import *
from data_utils import get_MNIST_data
from gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from solver import Solver

data = get_MNIST_data()

x_train = data['X_test']

print(x_train)