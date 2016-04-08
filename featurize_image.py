from lasagne.utils import floatX
import pandas as pd
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import skimage.transform
import os

mod = pickle.load(open('../vgg_cnn_s.pkl','r'))
MEAN_IMAGE = mod['mean image']
# LABELS = pickle.load(open('../veg_idx.pkl'))


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


def get_dat():

    # home = '/mnt/images'
    home = "/home/han/Documents/Github/urFarmer/images/"

    X = []
    y = []
    folder_list = os.listdir(home)
    LABELS = {}

    for i, folder in enumerate(folder_list):
        LABELS[folder] = i
        for fn in os.listdir('{}/{}'.format(home, folder)):
            _, im = preprocess_image('{}/{}/{}'.format(home, folder, fn))
            X.append(im)
            y.append(LABELS[folder])

    X = np.concatenate(X)
    y = np.array(y).astype('int32')


    return X, y, LABELS

if __name__ == '__main__':
    X, y, LABELS = get_dat()
