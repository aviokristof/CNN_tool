# CONFIG.INI DESCRIPTION

### Configuration of neural network structure:

image:

    - train_image: number of images from both datasets yes/no used to train
    - valid_image: number of images from both datasets yes/no used to validate
    - test_image: number of images from both datasets yes/no used to test model
    - width: width size of the images
    - height: height size of the images
    - color: number of color layers of the images
    - number_of_classes: number of clasees to identify [2 classes for binary classification in this example]
 
conv_layer:
 
    - number_of_conv_layers: number of convolutional layers used in training
    - filters: number of filters used in convolutional layer
    - kernel_size: size of filter kernel
    - strides: how much is kernel shifted in every iteration
    - padding: strides reduce the size of your input data for every layer padding if "same" padding prevent to change size of layer. Size of layer stay the same
    - batchnormal: batchnormalization layer
    - activ_func: type of an activation function used in convolutional layer
 
dense 
 
    - number_of_dense: number of dense layers used in training of neural network
    - dense_size: size of dense layer
    - batchnormal: batchnormalization layer
    - activ_func: activation function used in dense layer of deep neural network
 
dropout
 
    - dropout_rate: one in k inputs will be randomly excluded from each update cycle.Fraction of the input units to drop. Between 0 to 1.
 
output_activation_function
 
    - type_of_outpout_activation_function: activation function used in the last outpout layer
 
fit 
 
    - optimizer: type of an optimizer
    - learning_rate:
    - loss: The purpose of loss functions is to compute the quantity that a model should seek to minimize during training. (categoricalcrossentropy,binarycrossentropy)
    - metrics: A metric is a function that is used to judge the performance of your model. Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model.
    - batch_size: defines the number of samples that will be propagated through the network.
    - epochs: number of epochs used in training neural network
    - shuffle:
 
evaluate
 
    - batch_size: defines the number of samples that will be propagated through the network.
    - model_outpout_dir: name of an outpout /file.h5
    - class_1: name of data to train
    - class_2: name of data labeled as different than data to trained