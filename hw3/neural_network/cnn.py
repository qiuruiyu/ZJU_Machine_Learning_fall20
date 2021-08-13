import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # begin answer
        C, H, W = input_dim
        # W1 = np.random.normal(0, weight_scale, (input_dims, num_filters))
        W1 = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        b1 = np.zeros((num_filters))
        W2 = np.random.normal(0, weight_scale, (int(num_filters*H*W/4), hidden_dim))
        b2 = np.zeros((hidden_dim))
        W3 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b3 = np.zeros((num_classes))
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3
        # end answer

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        N = X.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        X = X.astype(self.dtype)

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # begin answer
        # forward part 
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache2 = affine_relu_forward(out, W2, b2)
        out, cache3 = affine_forward(out, W3, b3)     
        scores = out
        # end answer

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # begin answer
        loss, dx = softmax_loss(out, y)
        dx, grads['W3'], grads['b3'] = affine_backward(dx, cache3)
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, cache2)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache1)

        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2  
        grads['W3'] += self.reg * W3  

        # end answer
        return loss, grads



class ExpConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters1=64, num_filters2=128, filter_size=3,
                 hidden_dim=256, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # begin answer
        C, H, W = input_dim
        # W1 = np.random.normal(0, weight_scale, (input_dims, num_filters))
        W1 = np.random.normal(0, weight_scale, (num_filters1, C, filter_size, filter_size))
        b1 = np.zeros((num_filters1))
        W2 = np.random.normal(0, weight_scale, (num_filters2, num_filters1, filter_size, filter_size))
        b2 = np.zeros((num_filters2))
        # W3 = np.random.normal(0, weight_scale, (num_filters3, num_filters2, filter_size, filter_size))
        # b3 = np.zeros((num_filters3))
        W3 = np.random.normal(0, weight_scale, (int(num_filters2*H*W/16), hidden_dim))
        b3 = np.zeros((hidden_dim))        
        W4 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b4 = np.zeros((num_classes))
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3
        self.params['W4'] = W4
        self.params['b4'] = b4
        # self.params['W5'] = W5
        # self.params['b5'] = b5
        # end answer

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        N = X.shape[0]

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        # W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        X = X.astype(self.dtype)

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # begin answer
        # forward part 
        # out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache2 = conv_relu_pool_forward(out, W2, b2, conv_param, pool_param)
        # out, cache3 = conv_relu_pool_forward(out, W3, b3, conv_param, pool_param)
        out, cache3 = affine_relu_forward(out, W3, b3)
        out, cache4 = affine_forward(out, W4, b4)   
        scores = out
        # end answer

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # begin answer
        loss, dx = softmax_loss(out, y)
        dx, grads['W4'], grads['b4'] = affine_backward(dx, cache4)
        dx, grads['W3'], grads['b3'] = affine_relu_backward(dx, cache3)
        dx, grads['W2'], grads['b2'] = conv_relu_pool_backward(dx, cache2)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache1)
        # dx, grads['W1'], grads['b1'] = conv_relu_backward(dx, cache1)

        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2))
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2  
        grads['W3'] += self.reg * W3  
        grads['W4'] += self.reg * W4   
        # grads['W5'] += self.reg * W5

        # end answer
        return loss, grads

        