# **Image Classification**

Image classification tool trains convolutional neural network from two types of images. Images that we want to detect and 
images of different type than images that we want to detect.

## **What does the tool do?**

## **Description**

1. command takes two inputs yes/no directory. As an input tool takes paths to two directories and path were we want to save trained deep neural network. Directory with images that we want to detect (yes_image_dir), directory with images different than the images that we want to detect (no_images_dir) and target directory (target). By default target is cwd.

![Screenshot from 2023-02-18 20-38-37](https://user-images.githubusercontent.com/91827782/219903632-362dbf72-9055-405a-ac7c-7a133ef026a5.png)

2. preparing photos in directories to train, separate images to test/train/valid directories. First step is to create data tree with passed images in cwd. After data tree is created images are turned into an numpy array np.shape(number of images, width of image, height of image, colors of image) and labeled np.shape(number of images, number of labels) (more details in prepare_input.py image_data())

![Screenshot from 2023-02-18 20-39-42](https://user-images.githubusercontent.com/91827782/219903639-e11d5c71-b303-4557-831e-17c10cc84c9e.png)

3. creates convolutional neural network in the designed pattern with parameters from config.ini file. When input data are prepared the are passed to created convolutional neural network structure. Parameters of CNN are configurable in config.ini file.

![Structure](https://user-images.githubusercontent.com/91827782/219903643-9263b176-1494-4b6b-ba52-983597507567.jpg)

4. fitting trained neural network. Next step is to fit created model of CNN and save trained neural network in {name}.h5 file

![fit](https://user-images.githubusercontent.com/91827782/219903648-53894bf3-8c2b-43d6-839a-d4ba0a89be4a.jpg)

5. testing accuracy with test directory photos

.jpg TODO

6. printing results

.jpg TODO

## **Modules used in tool**

python modules/technologies used in tool
cv2 OpenCV

![image](https://user-images.githubusercontent.com/91827782/219903685-de5a2e6b-1e36-4327-a89f-a431c1d08028.png)

click

![image](https://user-images.githubusercontent.com/91827782/219903695-e10b8cfb-bac9-4998-9d5f-037ed94a56db.png)

from configparser import ConfigParser image logo

numpy image logo

![image](https://user-images.githubusercontent.com/91827782/219903714-bb8df901-d525-43e5-a83b-746d88f08494.png)


keras image logo + tensorflow

![image](https://user-images.githubusercontent.com/91827782/219903730-4b30fc25-ceb7-47d0-87b5-bd14a413b8c2.png)


matplotlib image logo

![image](https://user-images.githubusercontent.com/91827782/219903739-7eabc612-bf7a-48b2-bf13-5622f1b5cc02.png)


How to Install and Run the Project?

clone/copy -> run command Done

How to Use the Project

Add a License???

Badges?

## How to use the tool?

Step by step

clone repo

run it from repo dir where you cloned it

prepare two dir with images one with images that you want to train and one with images different than imaages to train

pass two prepared dirs

results .h5 and plots in cwd



For Linux OS

You need two directories with images. One directory with type of images that you want to detect and second directory with random type images.

In terminal use the command

Command:

create_CNN [dir_to_images_to_train] [dir_to_image_different_than_images_to_train]

As the result you will get trained deep neural network saved in {name}.h5 file in cwd of repository


CONFIG.INI DESCRIPTION

Configuration of neural network structure:

Neural network structure is configered by parameteres defined in config.ini file

There are seven types of congigration such as:

image:

    - train_image: number of images from both datasets yes/no used to train
    - valid_image: number of images from both datasets yes/no used to validate
    - test_image: number of images from both datasets yes/no used to test model
    - width: width size of the images
    - height: height size of the images
    - color: number of color layers of the images
    - number_of_classes: number of clasees to identify one for one for no in this example
 
conv_ layer:
 
    - number_of_conv_layers: number of conolutional layers used in training
    - filters: number of filters used in conolutional layer
    - kernel_size: size of filter kernel
    - strides:
    - padding:
    - batchnormal: batchnormalization layer
    - activ_func: type of an activation function used in convolutional layer
 
dense 
 
    - number_of_dense: number of dense layers used in training of neural network
    - dense_size: size of dense layer
    - batchnormal: batchnormalization layer
    - activ_func: activation function used in dense layer of deep neural network
 
dropo ut
 
    - dropout_rate:
 
outpo ut_activation_function
 
    - type_of_outpout_activation_function: activation function used in the last outpout layer
 
fit 
 
    - optimizer: type of an optimizer
    - leraning_rate:
    - loss: type of loss used in training neural network
    - metrics: type of metrics used in training neural network
    - batch_size:
    - epochs: number of epochs used in training neural network
    - shuffle:
 
evalu ate
 
    - batch_size:
    - model_outpout_dir: name of an outpout /file.h5
    - class_1: name of data to train
    - class_2: name of data labeled as different than data to trained


You need two directories with two types of images. One directory contains images to detect and second directory contains different types of images that are not images that we want to detect.


Structure of neural network and algorithm:

convolutional layer -> flatten layer -> dense layer -> dropout layer -> output dense layer -> output activation function -> to train

model fit -> model evaluate -> results show

preparing input images data frame (from directories) -> creating deep neural network structure -> training created neural network with prepared input images

As a user we are passing path to directories with yes/no images and setting up config.ini file. If we don't want to change deep neural network structure we can use deafult config.ini file.
output is saved in name.h5 file in current working directory 
