# **Image Classification**

Image classification tool trains convolutional neural network from two types of images. Images that we want to detect and 
images of different type than images that we want to detect.

**1. Description**
 
**2. How to use the tool** 

**3. Config.ini** 

**4. Python modules used in tool**

 

## **1. Description**

    1. Command
    2. Input images
    3. CNN structure
    4. Fitting model
    5. Testing accuracy4
    6. Results

**1.1 Command**

Command takes two inputs yes/no directory. As an input tool takes paths to two directories and path were we want to save trained deep neural network. Directory with images that we want to detect (yes_image_dir), directory with images different than the images that we want to detect (no_images_dir) and target directory (target). By default target is cwd.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220096494-3718b8ad-a7a3-43f4-beec-c774cc1fb977.png" />
  
  **Figure 1.1 Command**
</p>

**1.2 Input images** 

Preparing photos in directories to train, separate images to test/train/valid directories. First step is to create data tree with passed images in cwd. After data tree is created images are turned into an numpy array np.shape(number of images, width of image, height of image, colors of image) and labeled np.shape(number of images, number of labels) (more details in prepare_input.py image_data())


**1.3 CNN structure**

Creates convolutional neural network in the designed pattern with parameters from config.ini file. When input data are prepared the are passed to created convolutional neural network structure. Parameters of CNN are configurable in config.ini file.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220151177-7a29dd2c-8b03-4a4f-aae7-672938de1366.png" />
  
  **Figure 1.3 CNN structure**
</p>

**1.4  Fitting model**

Fitting trained neural network. Next step is to fit created model of CNN and save trained neural network in {name}.h5 file

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220151000-1f046e9d-4a1f-4f39-843c-23a29e9f90d3.png" />
  
  **Figure 1.4 Fit model structure**
</p>

**1.5  Testing accuracy**

Testing accuracy with test directory photos

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220184072-85ce4e30-c2d8-443d-8c99-1c29ddbdc2b2.png" />
  
  **Figure 1.5 Testing accuracy**
</p>

**1.6 Results**

Printing results

.jpg TODO

## **2. How to use the tool?**

Step by step

**2.1** Prepare two dir with images one with images that you want to train and one with images different than imaages to train

**2.2** Clone repository from github or download zip file.

**2.3** Open cloned directory and run in terminal "pip install .". Run command from description image1.1. To run tool succesfully you need python 3.10> and python modules from poin 4.

**2.4** Results name.h5 file saved in cwd. And plots are also saved.

## **3. Configuration file**

    1. image
    2. conv_layer
    3. dense
    4. dropout
    5. output_activation_function
    6. fit
    7. evaluate

**3.1 image**
descr
**3.2 conv_layer**
des
**3.3 dense**
des
**3.4 dropout**
des
**3.5 output_activation_function**
dse
**3.6 fit**
des
**3.7 evaluate**
des

## **4. Python modules used in tool**

    1. OpenCV
    2. Click
    3. Numpy
    4. Tensorflow/Keras
    5. Matplotlib

<p align="left">
  <img src="https://user-images.githubusercontent.com/91827782/220164949-89e622dd-2c9b-4f27-abf9-666e20337e74.png" />
  
  **Figure 4.1 OpenCv**
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/91827782/220095844-b8068bad-0730-4b0c-af7a-174ba9815e23.png" />
  
  **Figure 4.2 Click**
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/91827782/220095615-f2e30d6f-c937-4715-8edd-45cb59d40fd6.png" />
  
  **Figure 4.3 Numpy**
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/91827782/220095205-2aea96aa-9ffb-4932-8478-1c61ef0d9391.png" />
  
  **Figure 4.4 Tensorflow/Keras**
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/91827782/220095320-69566ac2-9096-4ed9-8c55-4c952b05f0e0.png" />
  
  **Figure 4.5 Matplotlib**
</p>





