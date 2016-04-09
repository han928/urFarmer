from lasagne.utils import floatX
import pandas as pd
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import skimage.transform
import os
from itertools import cycle



mod = pickle.load(open('./vgg_cnn_s.pkl','r'))
MEAN_IMAGE = mod['mean image']
HOME = '/mnt/images'
LABELS = {k: v for v, k in enumerate(os.listdir(HOME))}


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


def gen_dict():
    """
    generate a dictionary of generator for each image category
    """
    folder_list = os.listdir(HOME)
    gen_dict={}
    for folder in folder_list:
        gen_dict[folder] = img_gen(folder)

    return gen_dict

def img_gen(folder):
    """
    create generator for individual folders
    """
    f_list = os.listdir('{}/{}'.format(HOME, folder))

    for f in cycle(f_list):
        yield f


def batch_generator(gen_dict, batch_size=10):
    """
    iterate through gen_list and get images to fit
    """

    X = []
    y = []


    for folder, generator in gen_dict.iteritems():
        for i in xrange(batch_size):
            # get batch_size number of images generated from each folder
            fn = next(generator)
            try:
                _, im = preprocess_image('{}/{}/{}'.format(HOME, folder, fn))
                X.append(im)
                y.append(LABELS[folder])
            except:
                print 'something wrong with', folder, fn
    X = np.concatenate(X)
    y = np.array(y).astype('int32')


    return X, y






if __name__ == '__main__':
    gen_dict = gen_dict()
    X, y = batch_generator(gen_dict)
