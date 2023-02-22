# **Image Classification**

Image classification tool trains convolutional neural network from two types of images. Images that we want to detect and 
images of different type than images that we want to detect.

**1. Description**
 
**2. How to use the tool** 

**3. Config.ini** 

**4. Python modules used in tool**

 

## **1. Description**

  Tool creates input frame from two directories with images. Input frame is passed to Convolutional neiral network algorithm and creates neural network model. Model is fitted. Accuracy of fitted model is checked and results of model accuracy is printed in terminal.

    1. Command
    2. Input images
    3. CNN structure
    4. Fitting model
    5. Testing accuracy
    6. Validation and accuracy

**1.1 Command**

Command takes two inputs yes/no directory. As an input tool takes paths to two directories and path where we want to save trained deep neural network. Directory with images that we want to detect (yes_image_dir), directory with images different than the images that we want to detect (no_images_dir) and target directory (target). By default target is cwd.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220096494-3718b8ad-a7a3-43f4-beec-c774cc1fb977.png" />
  <p align = "center">
  <em>Figure [1.1] Command</em>
</p>

**1.2 Input images** 

Preparing images in directories to train, separate images to test/train/valid directories. First step is to create data tree with passed images in cwd. After data tree is created images are turned into an numpy array np.shape(number of images, width of image, height of image, colors of image) and labeled np.shape(number of images, number of labels) (more details in prepare_input.py image_data())



<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220193272-c491bb37-d892-4af9-8175-17f52586a8f1.png" />
  <p align = "center">
  <em>Figure [1.2] Created directory tree</em>
</p>

**1.3 CNN structure**

Creates convolutional neural network in the designed pattern with parameters from config.ini file. When input data are prepared they are passed to the created convolutional neural network structure. Parameters of CNN are configurable in config.ini file. Results are plotted on the chart.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220151177-7a29dd2c-8b03-4a4f-aae7-672938de1366.png" />
  <p align = "center">
  <em>Figure [1.3] CNN structure</em>
</p>

**1.4  Fitting model**

Fitting trained neural network. Next step is to fit created model of CNN and save trained neural network in {name}.h5 file

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220151000-1f046e9d-4a1f-4f39-843c-23a29e9f90d3.png" />
  <p align = "center">
  <em>Figure [1.4] Fit model structure</em>
</p>

**1.5  Testing accuracy**

Testing accuracy with test directory images.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220184072-85ce4e30-c2d8-443d-8c99-1c29ddbdc2b2.png" />
  <p align = "center">
  <em>Figure [1.5] Testing accuracy</em>
</p>

**1.6 Validation and accuracy**

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220735929-956e58b9-ca82-4916-999b-b529b269909a.png" />
  <p align = "center">
  <em>Figure [1.6] Model accuracy</em>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220735960-c9d3a27a-636d-4edc-9ed8-e8c02ec47e48.png" />
  <p align = "center">
  <em>Figure [1.7] Model loss</em>
</p>

## **2. How to use the tool?**

Step by step

**2.1** Prepare two dir with images one with images that you want to train and one with images different than images to train

**2.2** Clone repository from github or download .zip file.

**2.3** Open cloned directory and run in terminal "pip install .". Run command from description [Figure 1.1] To run tool succesfully you need python 3.10> and python modules from fourth point.

**2.4** Results.h5 file is saved in cwd by default.

## **3. Configuration file**

Neural network structure is configured by parameteres defined in config.ini file. More deatails in configuration.md file.

    1. image: parameteres used in preparing input frame
    2. conv_layer: parameters used in designing of convolutional layers
    3. dense: parameters used in designing of dense layers
    4. dropout: parameters used in designing dropout layer
    5. output_activation_function:parameters used in designing of output activation layer
    6. fit: parameters used in designing fitting model
    7. evaluate: parameters used in evaluating model

## **4. Python modules used in tool**

    1. OpenCV
    2. Click
    3. Numpy
    4. Tensorflow/Keras
    5. Matplotlib

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220164949-89e622dd-2c9b-4f27-abf9-666e20337e74.png" />
<p align = "center">
  <em>Figure [4.1] OpenCv</em>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220095844-b8068bad-0730-4b0c-af7a-174ba9815e23.png" />
<p align = "center">
  <em>Figure [4.2] Click</em>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220095615-f2e30d6f-c937-4715-8edd-45cb59d40fd6.png" />
<p align = "center">
  <em>Figure [4.3] Numpy</em>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220095205-2aea96aa-9ffb-4932-8478-1c61ef0d9391.png" />
<p align = "center">
  <em>Figure [4.4] Tensorflow/Keras</em>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/91827782/220095320-69566ac2-9096-4ed9-8c55-4c952b05f0e0.png" />
<p align = "center">
  <em>Figure [4.5] Matplotlib</em>
</p>