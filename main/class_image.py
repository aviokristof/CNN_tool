#class_image

import os
import shutil
import cv2

class Photo:
    '''
        Class Photo is used to copy images from directories and use them in training convolutional neural network.

        ---init(self,train_len,valid_len,test_len,width,height,color)
            define number of train images, images used in validation, images to test the result of trained deep neural network
            width of images, width of images, number of color layers of images (RGB =3 layers)
        ---get_len(self,dir)
        ---get_dir(self,dir)
        ---copy_image(self,input_file,name,flag)
        ---resize(self,dir)
        ---from_dir_to_structure(self,dir)


        
        '''

    def __init__(self,train_len,valid_len,test_len,width,height,color):
        '''Init Images source directory'''
        
        self.train_len = train_len
        self.valid_len = valid_len
        self.test_len = test_len
        self.width = width
        self.height = height
        self.color = color
        

    def get_len(self,dir):
        '''Returns number of files in directory'''
        length = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
        return length

    def get_dir(self,dir):
        '''Creates new directory from passed directory as argument'''
        new_dir = f"{dir}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        return new_dir


    def copy_image(self,input_file,name,flag):
        '''Copy images to tree directories, created in from_dir_to_structure(), from source directory as /{name}_{index}.jpg.'''

        dir_name = "class_trained" if flag == "yes" else "opposite_class"
        
        src_file = os.listdir(input_file)

        for index,file_name in enumerate(src_file,start=1):
            if index < self.train_len+1:
                dest_dir = f"{os.getcwd()}/train/{dir_name}/{name}_{index}.jpg"
            elif self.train_len-1 < index < self.train_len+self.valid_len+1:
                dest_dir = f"{os.getcwd()}/valid/{dir_name}/{name}_{index-self.train_len}.jpg"
            elif self.train_len+self.valid_len-1 < index <self.train_len+self.valid_len+self.test_len+1:
                dest_dir = f"{os.getcwd()}/test/{dir_name}/{name}_{index-self.train_len-self.valid_len}.jpg"
            else:
                break

            full_file_name = os.path.join(input_file,file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name,dest_dir)



    def resize(self,dir):
        '''If image is different size than we want changes size of an image in directory '''
        src_file = os.listdir(dir)
        for file in  src_file:
            img = cv2.imread(os.path.join(dir,file))
            image_100x100 =cv2.resize(img,(self.width,self.height))
            cv2.imwrite(os.path.join(dir,file),image_100x100)

    def from_dir_to_structure(self,dir):
        '''
        Creates tree of directories:
        ---dir passed as argument to function
            ---train
            ---valid
            ---test
                ---class_trained
                ---opposite_class

        '''
        inside_directories = ("class_trained","opposite_class")
        directories = ("train","valid","test")
        for first_dir in directories:
            path_to_check = os.path.join(dir,first_dir)
            if os.path.exists(path_to_check):
                #checks if dir (train,valid or test) exists, delete old_dir and create new
                shutil.rmtree(path_to_check)

            second_dir = self.get_dir(path_to_check)

            for x in inside_directories:
                new_dir = os.path.join(second_dir,x)
                self.get_dir(new_dir)

    


