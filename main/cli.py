#! /usr/bin/env python
import click
import os
from main import __version__
from main.prepare_input import CNN
from configparser import ConfigParser
import numpy as np
from keras.layers import Input

'''
Commannd line interface tool to train convolutional neural network.

create_CNN [path_1] [path_2] [target]

As an input takes path to directory with training images and path to directory with testing images. Creates folder structure:

---current_working_directory
    ---train
    ---valid
    ---test
        ---class_trained
        ---opposite_class

Splits images to train valid and test directory. Images from training directory copies to class_trained directory. Images from testing directory copies to opposite_class directory

Takes images and save them in the input frames and labels them as training_images and not_training images.
Input frame is passed as an argument to default or custom CNN architecture.
Deep neural network model is trained based on CNN architecture and input frame.

Trained CNN model is saved in target drectory (current directory by default)

Results of trained model are printed out

'''

def load_it(obj):
    '''
    Loading parameters from configuration file. Configparser reads every type of data as string. This function reads float,int,string
    and boolean parametere not only strings.
    '''
    if isinstance(obj, dict):
        return {k: load_it(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [load_it(elem) for elem in obj]
    if isinstance(obj, str):
        if obj == 'None':
            return None
        if obj.isnumeric():
            return int(obj)
        if obj.replace('.', '', 1).isnumeric():
            return float(obj)
        if obj.upper() in ('TRUE', 'FALSE', 'T', 'F'):
            return obj.upper() in ('TRUE', 'T')

    return obj

def load_config():
    '''
    Loading parameters from configuration file.
    '''
    config = ConfigParser()
    config.sections()
    config.read(f"{os.getcwd()}/config.ini")

    image = load_it(dict(config['image']))

    conv_layer = load_it(dict(config['conv_layer']))
    dense = load_it(dict(config['dense']))
    dropout = load_it(dict(config['dropout']))
    output_activation = load_it(dict(config['output_activation_function']))
    fit = load_it(dict(config['fit']))
    evaluate = load_it(dict(config['evaluate']))

    return image,conv_layer,dense,dropout,output_activation,fit,evaluate

def from_config(yes_image_input_dir,no_image_input_dir,target):
    '''
    Main core of app algorithm.
    1. Loading parameters from config file. load_config()
    2. Copying photos from yes_image_input directory and no_image_input directory to created directory tree. image_prepare()
    3. Creating input data format as: np.shape(number of photos,width,height,colors) from photos in created tree directory. CNN(Photo).image_data() 
    4. Preparing data_format to keras Input function. (labels to categorical, image data values divide by 255.0)
    5. Creating convolutional layers from prepared input data.
    6. Flattening convolutional layers.
    7. Creating Dense layers.
    8. Creating dropout function.
    9. Creating outpout dense layer.
    10. Creating outpout activation function.
    11. Fitting created convolutional neural network model.
    12. Evalueating created model.
    13. Printing results.
    '''
    #1.
    image,conv_layer,dense,dropout,output_activation,fit,evaluate = load_config()
    #2.
    data_format = image_prepare(image['number_of_classes'],no_image_input_dir,yes_image_input_dir,image['train_image'],image['valid_image'],image['test_image'],image['width'],image['height'],image['color'],target)
    #3.
    input_data_frame = data_format.image_data(os.getcwd())#directory same as setup.py file
    #4.
    input_data_frame = data_format.frame_categorical(input_data_frame) #to categorical

    input_shape = Input(shape = np.shape(input_data_frame["x_train"][0]))# 0 is to take shape (100,100,3) from (300,100,100,3)
    #5.
    input = input_shape
    for i in range(conv_layer['number_of_conv_layers']):
        input = data_format.get_convolutional_layer(input,conv_layer['filters'],conv_layer['kernel_size'],conv_layer['strides'],conv_layer['padding'],conv_layer['batchnormal'],conv_layer['activ_func'])
    #6.
    input = data_format.get_flatten(input)
    #7.
    for i in range(dense['number_of_dense']):
        input = data_format.get_dense(dense['dense_size'],input)
        if dense['batchnormal']:
            input = data_format.get_batchnormalization(input)
    
        input = data_format.get_activation_function(input,dense['activ_func'])
    #8.
    input = data_format.get_dropout(dropout['dropout_rate'],input)
    #9.
    input = data_format.get_output_dense(input)
    #10.
    output_layer = data_format.get_output_activation_function(input,output_activation['type_of_output_activ_function'])
    #11.
    model = data_format.model_fit(input_shape,output_layer,input_data_frame["x_valid"],input_data_frame["y_valid"],input_data_frame["x_train"],input_data_frame["y_train"],fit['learning_rate'],fit['loss'],fit['optimizer'],fit['metrics'],fit['batch_size'],fit['epochs'],fit['shuffle'])
    #12.
    evaluate,preds_single,actual_single = data_format.model_evaluate(model,input_data_frame["x_test"],input_data_frame["y_test"],evaluate['batch_size'],classes=[evaluate['class_1'],evaluate['class_2']],model_dir=f"{os.getcwd()}/{evaluate['model_output_dir']}")
    #13.
    click.echo(preds_single)
    click.echo(actual_single)
    result = data_format.results_printing(10,input_data_frame["x_test"],15,3,preds_single,actual_single)   


@click.command()
@click.argument("yes_image_input",type = click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True))
@click.argument("no_image_input",type = click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True))
@click.argument("target",type = click.Path(exists=True, file_okay=True, dir_okay=True, resolve_path=True),default = ".")



def create_CNN(yes_image_input,no_image_input,target):
    '''
    Cli main command: 'create_CNN [yes_image] [no_imageye]'
    '''


    from_config(yes_image_input,no_image_input,target)

         
def image_prepare(num_classes,no_image_input,yes_image_input,train_len,valid_len,test_len,width,height,color,target):#ad target
    '''
    Creating directory tree in cwd. Copying images from yes/no type photos folders and resizing images.
    '''
    
    import_image = CNN(num_classes,train_len,valid_len,test_len,width,height,color)
    import_image.from_dir_to_structure(target)

    import_image.copy_image(yes_image_input,"dron","yes")
    import_image.copy_image(no_image_input,"nie_dron","no")

    dir_1 = ("train","valid","test")
    dir_2 = ("class_trained","opposite_class")
    for name_1 in dir_1:
        for name_2 in dir_2:
            import_image.resize(f"{os.getcwd()}/{name_1}/{name_2}/")
    return import_image
