# **Image Classification**

Image classification tool trains convolutional neural network from two types of images. Images that we want to detect and 
images of different type than images that we want to detect.

## **Description**

1. Command takes two inputs yes/no directory. As an input tool takes paths to two directories and path were we want to save trained deep neural network. Directory with images that we want to detect (yes_image_dir), directory with images different than the images that we want to detect (no_images_dir) and target directory (target). By default target is cwd.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220096494-3718b8ad-a7a3-43f4-beec-c774cc1fb977.png" />
</p>


2. Preparing photos in directories to train, separate images to test/train/valid directories. First step is to create data tree with passed images in cwd. After data tree is created images are turned into an numpy array np.shape(number of images, width of image, height of image, colors of image) and labeled np.shape(number of images, number of labels) (more details in prepare_input.py image_data())

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220109434-ac982b97-cad7-4177-b4ca-4a80e383ba4b.png" />
</p>

3. Creates convolutional neural network in the designed pattern with parameters from config.ini file. When input data are prepared the are passed to 

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220151177-7a29dd2c-8b03-4a4f-aae7-672938de1366.png" />
</p>
4. Fitting trained neural network. Next step is to fit created model of CNN and save trained neural network in {name}.h5 file

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220151000-1f046e9d-4a1f-4f39-843c-23a29e9f90d3.png" />
</p>
5. testing accuracy with test directory photos

.jpg TODO

6. printing results

.jpg TODO

## **Modules used in tool**

![image](https://user-images.githubusercontent.com/91827782/220152612-f0039670-70d5-49a6-bb1c-e6dcb298b0e5.png)

![image](https://user-images.githubusercontent.com/91827782/220095844-b8068bad-0730-4b0c-af7a-174ba9815e23.png)

![image](https://user-images.githubusercontent.com/91827782/220095615-f2e30d6f-c937-4715-8edd-45cb59d40fd6.png)

![image](https://user-images.githubusercontent.com/91827782/220095205-2aea96aa-9ffb-4932-8478-1c61ef0d9391.png)

![image](https://user-images.githubusercontent.com/91827782/220095320-69566ac2-9096-4ed9-8c55-4c952b05f0e0.png)


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
