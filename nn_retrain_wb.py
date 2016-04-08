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
import cPickle as pickle
import matplotlib.pyplot as plt
import skimage.transform
import os


# Without GPU
from lasagne.layers import Conv2DLayer as ConvLayer
# With GPU
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer


mod = pickle.load(open('vgg_cnn_s.pkl','r'))
params = mod['values']
MEAN_IMAGE = mod['mean image']
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
                        # ('fc3', DenseLayer),
                        # ('drop3', DropoutLayer),
                        ('output', DenseLayer)
               ],

               # Input Layer
               input_shape=(None, 3, 224, 224),

               # Conv1
               conv1_num_filters=96,
               conv1_filter_size=7,
               conv1_stride=2,
               conv1_flip_filters=False,
               conv1_w=params[0],
               conv1_b=params[1],

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
               conv2_w=params[2],
               conv2_b=params[3],

               # pool2
               pool2_pool_size=2,
               pool2_stride=2,
               pool2_ignore_border=False,

               # conv3
               conv3_num_filters=512,
               conv3_filter_size=3,
               conv3_pad=1,
               conv3_flip_filters=False,
               conv3_w=params[4],
               conv3_b=params[5],

               # conv4
               conv4_num_filters=512,
               conv4_filter_size=3,
               conv4_pad=1,
               conv4_flip_filters=False,
               conv4_w=params[6],
               conv4_b=params[7],


               # conv5
               conv5_num_filters=512,
               conv5_filter_size=3,
               conv5_pad=1,
               conv5_flip_filters=False,
               conv5_w=params[8],
               conv5_b=params[9],

               # pool3
               pool3_pool_size=3,
               pool3_stride=3,
               pool3_ignore_border=False,

               # fc1
               fc1_num_units=4096,
               fc1_nonlinearity=rectify,
               fc1_w=params[10],
               fc1_b=params[11],

               # DropoutLayer1
               drop1_p = 0.5,

               # fc2
               fc2_num_units=4096,
               fc2_nonlinearity=rectify,
               fc2_w=params[12],
               fc2_b=params[13],

               # DropoutLayer2
               drop2_p = 0.5,

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



def preprocess_image(im_file):
    """
    preprocess image to 256 for neural network to work
    """
    # ways to get image on the web
    # import io
    # ext = url.split('.')[-1]
    # im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)

    im = plt.imread(open(im_file, 'r'))

    # resize to smalled dimension of 256 while preserving aspect ratio
    h, w, c = im.shape

    if h < w:
        im = skimage.transform.resize(im, (256, w/h*256), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h/w*256, 256), preserve_range=True)

    h, w, c = im.shape

    # central crop to 224x224
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE
    return rawim, floatX(im[np.newaxis])


def generate_feature(folder):
    # home = '/mnt/images/'
    home = "/home/han/Documents/Github/urFarmer/images/"
    try:
        print "filepath", home+folder
        file_list = os.listdir(home+folder)
    except:
        print "no such file"
        return ""

    for i, f in enumerate(file_list):
        fname = home+folder+'/'+f
        _, im = preprocess_image(fname)

        if i ==0:
            input_im = im
        else:
            input_im = np.r_[input_im, im]

        # save every 300 images to prevent overflow
        # if i % 300 == 0 and i != 0:
        #     out = nn.predict(input_im)
        #
        #     np.save(folder+str(i)+'.npy', out)


    # out = nn.predict(input_im)


    # print out
    np.save(folder + '.npy', input_im)


def folder_gen_feature():
    # home = '/mnt/images/'
    home = "/home/han/Documents/Github/urFarmer/images/"



    folder_list = os.listdir(home)

    for folder in folder_list:
        try:
            generate_feature(folder)
        except:
            print "something wrong with ", folder


if __name__ == '__main__':
    folder_gen_feature()
