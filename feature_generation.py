
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
import sys, os





# Without GPU
from lasagne.layers import Conv2DLayer as ConvLayer
# With GPU
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer


class NN_1(object):
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
        output_layer = net['fc7']


        lasagne.layers.set_all_param_values(output_layer, self.model['values'][:-2])

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
        out = np.array(lasagne.layers.get_output(self.output_layer, im, deterministic=True).eval())
        return out



def generate_feature(folder):
    """
    transform image to feature using pretrained neural network and save as numpy
    """

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
        _, im = nn.preprocess_image(fname)

        if (i ==0 or i%300==1) and i != 1:
            input_im = im
        else:
            input_im = np.r_[input_im, im]

        # save every 300 images to prevent overflow
        if i % 300 == 0 and i != 0:
            out = nn.predict(input_im)

            np.save(folder+str(i)+'.npy', out)


    out = nn.predict(input_im)


    # print out
    np.save(folder + '.npy', out)


def folder_gen_feature():
    """
    helper function to get a folders with images and transfom them to numpy array
    features
    """
    # home = '/mnt/images/'
    home = "/home/han/Documents/Github/urFarmer/images/"



    folder_list = os.listdir(home)

    for folder in folder_list:
        try:
            generate_feature(folder)
        except:
            print "something wrong with ", folder


if __name__ == "__main__":
    # folder = sys.argv[1]
    nn = NN_1()
    nn.build()
    import time
    t1 = time.time()
    folder_gen_feature()
    t2 = time.time()
    print "took ", t2-t1
    # generate_feature(folder)
