from lasagne import layers
from lasagne.nonlinearities import  sigmoid, softmax, rectify, tanh, linear
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from lasagne.layers import Conv2DLayer as ConvLayer
import pandas as pd
import numpy as np

def nn_set():
    """
    # Configure the neural network
    """
    nn = NeuralNet(
               # Specify the layers
               layers=[('input', InputLayer),
                        ('drop1', DropoutLayer),
                       ('hidden1', layers.DenseLayer),
                       ('output', layers.DenseLayer),
               ],

               # Input Layer
               input_shape=(None, 4096),

               # DropoutLayer
               drop1_p = 0.5,

               # 1st Hidden Layer
               hidden1_num_units=4096,
               hidden1_nonlinearity=rectify,

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

def load():
    """
    load file from pickle, target is vegetable names and label is number
    """
    df = pd.read_pickle('training.pkl')

    X = df.drop(['target', 'label'], axis=1).values
    y = df['label'].values
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    return X, y

def top_5_error(X_test, y_test):
    top5 = 0.

    prob = nn.predict_proba(X_test)
    for i, cat in enumerate(y_test):
        if cat in np.argsort(prob[i])[::-1][:5]:
            top5+=1

    return top5/y_test.shape[0]


if __name__ == '__main__':
    X, y = load()
    nn = nn_set()

# save params get_all_param_values()
# load param load_param_from


# 50.5% nesterov_momentum update rate0.002 ,  77.6 top 5 error rate
# 51.1% add dropout layer  79.29% top 5 error rate
