import numpy as np
import matplotlib.pyplot as plt
import os
from main.class_image import Photo
import re
import logging
#import class_image

from keras.utils import to_categorical
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout,LeakyReLU, Activation,ReLU,ELU 
from keras.models import Model, save_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from keras.utils.image_utils import load_img,img_to_array


class CNN(Photo):

    def __init__(self,num_classes,train_len,valid_len,test_len,width,height,color):
        super().__init__(train_len,valid_len,test_len,width,height,color)
        self.current_dir = os.getcwd()
        self.num_classes = num_classes

        


    
    def get_convolutional_layer(self,input_layer,filters,kernel_size,strides,padding,batchnormal,activ_func):
        '''
        Creates tf.keras convolutional layer with batchnormalization if TRUE and activation function: relu,elu,leakyrelu.
        '''
       
        conv_layer = Conv2D(
            filters= filters,
            kernel_size = (kernel_size,kernel_size) ,
            strides= strides,
            padding=padding
        )(input_layer)
        if batchnormal:
            conv_layer = self.get_batchnormalization(conv_layer)
        if activ_func:
            conv_layer = self.get_activation_function(conv_layer,activ_func)

        return conv_layer
    
    def get_batchnormalization(self,input):
        '''
        Creates tf.keras batchnormalization layer.
        '''
        
        batch = BatchNormalization()(input)

        return batch

    def get_activation_function(self,input,name):
        '''
        Creates tf.keras activation function layer:
            -leakyrelu or
            -relu or
            -elu 
        '''
        #default leakyrelu, pass type of active func as parmeter
        if name == 'leakyrelu':
            activ_func = LeakyReLU()(input)
        elif name == 'relu':
            activ_func = ReLU()(input)
        elif name == 'elu':
            activ_func = ELU()(input)
        else:
            logging.error('Choose one of: "leakyrelu" or "relu" or "elu"')
        return activ_func
    
    def get_flatten(self,input):
        '''
        Creates tf.keras flatten layer
        '''

        flatten_layer = Flatten()(input)

        return flatten_layer

    def get_dense(self,size,input):
        '''
        Creates tf.keras dense layer
        '''
        if type(size) == int:
            dense = Dense(size)(input)
        else:
            logging.error('Size has to be an int')

        return dense
    
    def get_output_dense(self,input):

        '''
        Creates tf.keras output dense layer
        Size of this layer has to have length equal to number of CNN.num.classes 
        '''
        
        if type(self.num_classes) == int:
             
            out_dense = Dense(self.num_classes)(input)
        else:
            logging.error('Size has to be an int')
        return out_dense

    def get_dropout(self,rate,input):
        '''
        Creates tf.keras dropout layer.
        '''

        if rate >= 0.0:

            dropout = Dropout(rate)(input)
        else:
            logging.error('Rate has to be float and more than 0')

        return dropout

    def get_output_activation_function(self,input,name):

        '''
        Creates tf.keras output activation function.
        For binary classification should be sigmoid.
        Output activation function for recognizing types of images has to be softmax function.
        '''
        if name == 'softmax':
            activ_func = Activation(name)(input)
        elif name == 'sigmoid':
            activ_func = Activation(name)(input)
        else:
            logging.error('Choose betweeen: "softmax" or "sigmoid"')

        
        return activ_func

    def model_evaluate(self,model,x_test,y_test,batch_size,classes,model_dir):
        '''
        Evaluating tf.keras model created and trained in model_fit() function. Saving model to f"{model_dir}/file.h5".
        '''
        model.evaluate(x_test,y_test,batch_size,verbose=2)
        classes = np.array(classes)

        preds = model.predict(x_test)
        preds_single = classes[np.argmax(preds,axis= -1)]
        actual_single = classes[np.argmax(y_test, axis = -1)]

        model_saved = save_model(model,model_dir,save_format='h5')

        return model_saved,preds_single,actual_single
    
    def results_printing(self,to_show,x_test,figsize_h,figsize_w,preds_single,actual_single):
        '''
        Printing results of classification. Change this.
        '''
        #model_saved print
        indices = np.random.choice(range(len(x_test)),to_show)
        fig = plt.figure(figsize=(figsize_h,figsize_w))
        fig.subplots_adjust(hspace=0.4,wspace=0.4)

        for index,image in enumerate(indices):
            img = x_test[image]
            ax = fig.add_subplot(1,to_show, index+1)
            ax.text(0.5, -0.35, f'pred = {preds_single[image]}', fontsize=10, ha='center', transform=ax.transAxes) 
            ax.text(0.5, -0.7, f'act = {actual_single[image]}', fontsize=10, ha='center', transform=ax.transAxes)
            ax.imshow(img)

    


    def image_data(self,input_dir):
        '''
        Every image is represented as 3 dimensional tensor. Width,height and depth. Width and height are dimensions of image.
        Depth is the number of color layers used to creating an image. By default colored images are created by Red, Green and Blue
        layers. Every pixel of a RGB image is represented as 3 dimensional array with density of red blue and green color.
        Range of pixels is between 0 and 255. (0,0,0)=black (255,255,255)=white.

        This function creates input data from images and represent them in defined data_structure such as:
            x_frame = np.shape(number of images, width of image, height of image, colors of image)
            example: x_frame = np.shape(100,50,50,3)
            
        This function also creates labels to categorize images:
            y_array = np.shape(number of images, number of labels)
            exapmle: y_frame = np.shape(2000,2)

        Output data structure is dictionary:
            -x_train:
            -y_train:
            -x_valid:
            -y_valid:
            -x_test:
            -y_test:

        '''

        data_iterator = {"train":self.train_len,"valid":self.valid_len,"test":self.test_len}

        data_frame = {"x_train":None,"y_train":None,"x_valid":None,"y_valid":None,"x_test":None,"y_test":None}

        for key,value in data_iterator.items():
            home_dir = os.path.join(input_dir,key)
            directory = os.listdir(home_dir)

            x_array = np.empty((0,self.width,self.height,self.color))

            for dir in directory:
                dir_to_loop = os.path.join(home_dir,dir)
                image_list = os.listdir(dir_to_loop)#not sorted images
                image_list_sorted = sorted(image_list,key=lambda f: int(re.sub('\D', '', f)))

                for index,image in enumerate(image_list_sorted):
                    x = load_img(os.path.join(dir_to_loop,image))
                    x = img_to_array(x)
                    x = x[np.newaxis,:]
                    x_array = np.append(x_array,x,axis=0)


            x_array = x_array.astype('uint8')
            y_yes = np.full((value,1),0,dtype='uint8')
            y_no = np.full((value,1),1,dtype='uint8')   
            y_array = np.append(y_yes,y_no,axis=0)
            data_frame[f'x_{key}'] = x_array
            data_frame[f'y_{key}'] = y_array


            

        return data_frame
        
    def frame_categorical(self,input_frame):
        '''
        Dividing x_frame values by 255.0 because colors of pixels are from 0 to 255. Value of every pixel represented by 
        red,blue,green color has to be between 0 to 1. Values are saved as 'float32' type.
        Labels have to be changed to categorical values.
        '''
        
        for key,value in input_frame.items():
            if 'x_' in key:
                input_frame[key] = value.astype('float32')/255.0
            elif 'y_' in key:
                input_frame[key] = to_categorical(value,self.num_classes)
        return input_frame

    def model_fit(self,input_layer,output_layer,x_valid,y_valid,x_train,y_train,learning_rate,loss,optimizer,metrics,batch_size,epochs,shuffle):
        #by default Adam later settable
        '''
        Training model with tf.keras fit() function.
        Optimizer used in this example is Adam(). Can be changed.
        '''
   
        model = Model(input_layer,output_layer)

        model.summary()
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            logging.error('Choose one of optimizers: "adam"')
        model.compile(loss=loss, optimizer=opt, metrics=[metrics])
        model.fit(x_train,
                y_train
                , batch_size
                , epochs
                , shuffle
                ,validation_data=(x_valid,y_valid)
                

                )

        model.layers[6].get_weights()
        return model

