import lasagne
from lasagne import layers
from lasagne.nonlinearities import  sigmoid, softmax, rectify, tanh, linear
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
import pandas as pd
import numpy as np

# Without GPU
from lasagne.layers import Conv2DLayer as ConvLayer
# With GPU
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer


def nn_set():
    """
    # Configure the neural network
    """
    nn = NeuralNet(
               # Specify the layers
               layers=[ ('input', InputLayer),
                        ('conv1', ConvLayer),
                        ('norm1', NormLayer),
                        ('pool1', PoolLayer),
                        ('conv2', ConvLayer),
                        ('pool2', PoolLayer),
                        ('conv3', ConvLayer),
                        ('conv4', ConvLayer),
                        ('conv5', ConvLayer),
                        ('pool3', PoolLayer),
                        ('fc1', DenseLayer),
                        ('drop1', DropoutLayer),
                        ('fc2', DenseLayer),
                        ('drop2', DropoutLayer),
                        ('fc3', DenseLayer),
                        ('drop3', DropoutLayer),
                        ('output', DenseLayer)
               ],

               # Input Layer
               input_shape=(None, 3, 224, 224),

               # Conv1
               conv1_num_filters=96,
               conv1_filter_size=7,
               conv1_stride=2,
               conv1_flip_filters=False,

               # norm1
               norm1_alpha=0.0001,

               # pool1
               pool1_pool_size=3,
               pool1_stride=3,
               pool1_ignore_border=False,


               # conv2
               conv2_num_filters=256,
               conv2_filter_size=5,
               conv2_flip_filters=False,

               # pool2
               pool2_pool_size=2,
               pool2_stride=2,
               pool2_ignore_border=False,

               # conv3
               conv3_num_filters=512,
               conv3_filter_size=3,
               conv3_pad=1,
               conv3_flip_filters=False,

               # conv4
               conv4_num_filters=512,
               conv4_filter_size=3,
               conv4_pad=1,
               conv4_flip_filters=False,


               # conv5
               conv5_num_filters=512,
               conv5_filter_size=3,
               conv5_pad=1,
               conv5_flip_filters=False,

               # pool3
               pool3_pool_size=3,
               pool3_stride=3,
               pool3_ignore_border=False,

               # fc1
               fc1_num_units=4096,
               fc1_nonlinearity=rectify,

               # DropoutLayer1
               drop1_p = 0.5,

               # fc2
               fc2_num_units=4096,
               fc2_nonlinearity=rectify,

               # DropoutLayer2
               drop2_p = 0.5,

               # fc3
               fc3_num_units=4096,
               fc3_nonlinearity=rectify,

               # DropoutLayer3
               drop3_p = 0.5,

               # Output Layer
               output_num_units=130,
               output_nonlinearity=softmax,

               # Optimization
               update=nesterov_momentum,
               update_learning_rate=0.05,
               update_momentum=0.7,
               max_epochs=30,

               # Others
               regression=False,
               verbose=1,
         )
    return nn
