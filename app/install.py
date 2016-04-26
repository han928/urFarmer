import os


pkl_dict = {
    'vgg_cnn_weight': "https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl",
    'urFarmer_weight': "https://s3.amazonaws.com/han.indiv.bucket2/good_params.pkl",
    'veg_index': "https://s3.amazonaws.com/han.indiv.bucket2/veg_idx.pkl"
}


for key, addr in pkl_dict.iteritems():
    print "=" * 20 + '\n'
    print "Downloading {}".format(key)
    print '\n' + "=" * 20 + '\n'

    os.system('wget {}'.format(addr))


print "Download complete, please run app.py in terminal to start the web app"
