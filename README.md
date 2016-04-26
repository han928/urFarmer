# urFarmer


## Motivation:
Have you ever went to a farmers market and wondering what vegetable/ingredient
is that and how to cook it? I cook a lot and that's the first questions I have
when I see a unknown vegetables or ingredients and that's the first questions
popping in my mind. While googling nowadays is easy to do on your phone, if you
don't know the name of that vegetable you probably not gonna find out about it. In this project, I aim to use machine learning to solve this problem.


## Objective:
In order to solve the aforementioned problems, I use a pretrained neural network
trained for Imagenet Large Scale image recognition task from [Oxford Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/practicals/cnn/#vgg-convolutional-neural-networks-practical) and take advantage of the layers before the output layer to featurize the input images than perform the image recognition task with 2 new DenseLayer.

## Methodology
In order to solve this problem using object recognition
using neural networks. While neural networks on ILSVRC challenges on Image-net have been trained to classify 1000 objects. The number of class in vegetables and fruits is less than 30 classes. On the other hand, to train an effected neural network for class recognition, the training of the neural network is very time consuming. By using the layer output before the output layer, we make use of a pretrained neural network to featurize our images and perform classification task from there.


## Data Source:
1. [imagenet data source](http://image-net.org/): a bunch of labeled, boxed datasets for images (eg. collard greens, brocolli)
2. [food2fork api](http://food2fork.com/about/api): for getting recipes after recognition



## Web App:

1. Run install.py by typing "python install.py" in the terminal to download essential files for running the web app

2. Go to [food2fork.com](http://food2fork.com/) to apply for an API key and replace your API key in food2fork_api_key.json files

3. run app.py and it should be working. It your machine does not have a GPU, please import Conv2DLayer Layer and commented out  the import for Conv2DDNNLayer 
