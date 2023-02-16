# Binary images Classification

Binary images classification tool trains convolutional neural network from two types of images. Images that we want to detect and 
images of different type than images that we want to detect.

## How to use the tool?

For Linux OS

You need two directories with images. One directory with type of images that you want to detect and second directory with random type images.

In terminal use the command

Command:

create_CNN [dir_to_images_to_train] [dir_to_image_different_than_images_to_train]

As the result you will get trained deep neural network saved in {name}.h5 file in cwd of repository

Configuration of neural network structure:

Neural network structure is configered by parameteres defined in config.ini file

There are seven types of congigration such as:
image:
    -train_image: number of images from both datasets yes/no used to train
    -valid_image: number of images from both datasets yes/no used to validate
    -test_image: number of images from both datasets yes/no used to test model
    -width: width size of the images
    -height: height size of the images
    -color: number of color layers of the images
    -number_of_classes: number of clasees to identify one for one for no in this example
conv_layer:
    -number_of_conv_layers: number of conolutional layers used in training
    -filters: number of filters used in conolutional layer
    -kernel_size: size of filter kernel
    -strides:
    -padding:
    -batchnormal: batchnormalization layer
    -activ_func: type of an activation function used in convolutional layer
dense
    -number_of_dense: number of dense layers used in training of neural network
    -dense_size: size of dense layer
    -batchnormal: batchnormalization layer
    -activ_func: activation function used in dense layer of deep neural network
dropout
    -dropout_rate:
outpout_activation_function
    -type_of_outpout_activation_function: activation function used in the last outpout layer
fit
    -optimizer: type of an optimizer
    -leraning_rate:
    -loss: type of loss used in training neural network
    -metrics: type of metrics used in training neural network
    -batch_size:
    -epochs: number of epochs used in training neural network
    -shuffle:
evaluate
    -batch_size:
    -model_outpout_dir: name of an outpout /file.h5
    -class_1: name of data to train
    -class_2: name of data labeled as different than data to trained


You need two directories with two types of images. One directory contains images to detect and second directory contains different types of images that are not images that we want to detect.


Structure of neural network and algorithm:

convolutional layer -> flatten layer -> dense layer -> dropout layer -> output dense layer -> output activation function -> to train

model fit -> model evaluate -> results show

preparing input images data frame (from directories) -> creating deep neural network structure -> training created neural network with prepared input images

As a user we are passing path to directories with yes/no images and setting up config.ini file. If we don't want to change deep neural network structure we can use deafult config.ini file.
output is saved in name.h5 file in current working directory 