# have to start environment on my pc: source activate neural_net_env
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import skimage.transform
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

# Without GPU
from lasagne.layers import Conv2DLayer as ConvLayer
# With GPU
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer


class NN(object):
    def __init__(self):
        # import pretrained weights
        self.model = pickle.load(open('vgg_cnn_s.pkl'))
        self.CLASSES = np.array(self.model['synset words'])
        self.MEAN_IMAGE = self.model['mean image']


    def build(self):
        """
        build neural net model and set weight as pretrained weight from lasagne modelzoo
        """
        # set layers
        net = {}
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
        net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
        net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
        net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
        net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
        net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
        net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
        net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
        net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
        net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
        net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
        net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
        output_layer = net['fc8']


        lasagne.layers.set_all_param_values(output_layer, self.model['values'])

        self.output_layer = output_layer


    def preprocess_image(self, im_file):
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

        im = im - self.MEAN_IMAGE
        return rawim, floatX(im[np.newaxis])


    def predict(self, im):
        """
        Predict outcome of a image
        """
        probs = np.array(lasagne.layers.get_output(self.output_layer, im, deterministic=True).eval())
        # minus the probability to have reverse order
        cats = self.CLASSES[np.argsort(-probs)]

        # return np.array([result for result in zip(cats.flatten(), probs.flatten())])
        return cats[0]
