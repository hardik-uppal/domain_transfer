from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import h5py
import cv2
import os
import random
from keras.utils.np_utils import to_categorical
import matplotlib.pylab as plt
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model
import keras.backend as K
from keras.layers import Input, Concatenate, concatenate
import keract
#import model.models as model_lib


def normalization(X):
    result = X / 127.5 - 1
    
    # Deal with the case where float multiplication gives an out of range result (eg 1.000001)
    out_of_bounds_high = (result > 1.)
    out_of_bounds_low = (result < -1.)
    out_of_bounds = out_of_bounds_high + out_of_bounds_low
    
    if not all(np.isclose(result[out_of_bounds_high],1)):
        raise RuntimeError("Normalization gave a value greater than 1")
    else:
        result[out_of_bounds_high] = 1.
        
    if not all(np.isclose(result[out_of_bounds_low],-1)):
        raise RuntimeError("Normalization gave a value lower than -1")
    else:
        result[out_of_bounds_low] = 1.
    
    return result

def get_optimizer(opt, lr):

    if opt == "SGD":
        return SGD(lr=lr)
    elif opt == "RMSprop":
        return RMSprop(lr=lr)
    elif opt == "Adam":
        return Adam(lr=lr, beta1=0.5)

def inverse_normalization(X):
    # normalises back to ints 0-255, as more reliable than floats 0-1
    # (np.isclose is unpredictable with values very close to zero)
    result = ((X + 1.) * 127.5).astype('uint8')
    # Still check for out of bounds, just in case
    out_of_bounds_high = (result > 255)
    out_of_bounds_low = (result < 0)
    out_of_bounds = out_of_bounds_high + out_of_bounds_low
    
    if out_of_bounds_high.any():
        raise RuntimeError("Inverse normalization gave a value greater than 255")
        
    if out_of_bounds_low.any():
        raise RuntimeError("Inverse normalization gave a value lower than 1")
        
    return result


def get_nb_patch(img_dim, patch_size, image_data_format):

    assert image_data_format in ["channels_first", "channels_last"], "Bad image_data_format"

    if image_data_format == "channels_first":
        assert img_dim[1] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[2] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[1] // patch_size[0]) * (img_dim[2] // patch_size[1])
        img_dim_disc = (img_dim[0], patch_size[0], patch_size[1])

    elif image_data_format == "channels_last":
        assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
        assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
        nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
        img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, image_data_format, patch_size):

    # Now extract patches form X_disc
    if image_data_format == "channels_first":
        X = X.transpose(0,2,3,1)

    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])

    if image_data_format == "channels_first":
        for i in range(len(list_X)):
            list_X[i] = list_X[i].transpose(0,3,1,2)

    return list_X


def load_data(dset, image_data_format):

    with h5py.File("../../data/processed/%s_data.h5" % dset, "r") as hf:

        X_full_train = hf["train_data_full"][:].astype(np.float32)
        X_full_train = normalization(X_full_train)

        X_sketch_train = hf["train_data_sketch"][:].astype(np.float32)
        X_sketch_train = normalization(X_sketch_train)

        if image_data_format == "channels_last":
            X_full_train = X_full_train.transpose(0, 2, 3, 1)
            X_sketch_train = X_sketch_train.transpose(0, 2, 3, 1)

        X_full_val = hf["test_data_full"][:].astype(np.float32)
        X_full_val = normalization(X_full_val)

        X_sketch_val = hf["test_data_sketch"][:].astype(np.float32)
        X_sketch_val = normalization(X_sketch_val)

        if image_data_format == "channels_last":
            X_full_val = X_full_val.transpose(0, 2, 3, 1)
            X_sketch_val = X_sketch_val.transpose(0, 2, 3, 1)

        return X_full_train, X_sketch_train, X_full_val, X_sketch_val


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx]

def gen_batch_cat(X1, X2, batch_size):
    counter = int(X1.shape[0]/batch_size)
    while counter>0:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        counter = counter - 1
        yield X1[idx], X2[idx]

def sample_noise(noise_scale, batch_size, noise_dim):

    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0],noise_dim[1],noise_dim[2]))


def get_disc_batch_wgan(X_real_batch, X_rgb_batch, generator_model, batch_counter, batch_size, patch_size, image_data_format, noise_dim, noise_scale=0.5):

    # Pass noise to the generator
#    noise_input = sample_noise(noise_scale, batch_size, noise_dim)
    # Produce an output
    X_disc_gen, pred_class = generator_model.predict([X_rgb_batch])
#    X_disc_real = X_real_batch
#    y_disc_real = -np.ones(X_disc_real.shape[0])
#    y_disc_gen = np.ones(X_disc_real.shape[0])
    
    # Now extract patches form X_disc
    X_disc_gen = extract_patches(X_disc_gen, image_data_format, patch_size) 
    X_disc_real = extract_patches(X_real_batch, image_data_format, patch_size) 

    return X_disc_real, X_disc_gen

def get_disc_batch(X_full_batch, X_sketch_batch, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    X_disc = generator_model.predict(X_sketch_batch)
    
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
#        y_disc[:, 0] = 1

#        if label_flipping > 0:
#            p = np.random.binomial(1, label_flipping)
#            if p > 0:
#                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch
        y_disc = np.ones((X_disc.shape[0], 1), dtype=np.uint8)
#        if label_smoothing:
#            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
#        else:
#            y_disc[:, 1] = 1

#        if label_flipping > 0:
#            p = np.random.binomial(1, label_flipping)
#            if p > 0:
#                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc

def get_disc_batch_cat(X_full_batch, X_sketch_batch, y_label, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):
    
    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        y_filler = np.zeros((X_disc.shape[0], 52), dtype=np.uint8)
        
        y_disc = np.concatenate((y_label, y_filler), axis=1)
#        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
#        y_disc[:, 0] = 1

#        if label_flipping > 0:
#            p = np.random.binomial(1, label_flipping)
#            if p > 0:
#                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch
        y_filler = np.zeros((X_disc.shape[0], 52), dtype=np.uint8)
        
        y_disc = np.concatenate((y_filler, y_label), axis=1)
#        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
#        if label_smoothing:
#            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
#        else:
#            y_disc[:, 1] = 1
#
#        if label_flipping > 0:
#            p = np.random.binomial(1, label_flipping)
#            if p > 0:
#                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, image_data_format, patch_size)

    return X_disc, y_disc


def get_batch_cnn(X_full_batch, X_sketch_batch, y_label, generator_model, batch_counter, patch_size,
                   image_data_format, label_smoothing=False, label_flipping=0):
    
    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_sketch_batch)
        y_filler = np.zeros((X_disc.shape[0], 52), dtype=np.uint8)
        
        y_disc = np.concatenate((y_label, y_filler), axis=1)
#        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
#        y_disc[:, 0] = 1

#        if label_flipping > 0:
#            p = np.random.binomial(1, label_flipping)
#            if p > 0:
#                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_full_batch
        y_filler = np.zeros((X_disc.shape[0], 52), dtype=np.uint8)
        
        y_disc = np.concatenate((y_filler, y_label), axis=1)
#        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
#        if label_smoothing:
#            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
#        else:
#            y_disc[:, 1] = 1
#
#        if label_flipping > 0:
#            p = np.random.binomial(1, label_flipping)
#            if p > 0:
#                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    return X_disc, y_disc

class data_pipeline:
    
    def __init__(self, image_dir, dataset, test_train=None):
        self.image_full_arr = []
        self.label_full_arr = []
#        self.batch_size = batch_size
        self.dataset = dataset
        self.test_train = test_train
        #check if input dir is list or not; create two lsit with addresses and corresponding labels
        if isinstance(image_dir, list):
            for image_dir_temp in image_dir:
                label_list = os.listdir(image_dir_temp)
                
                for label in label_list:
                    label_dir = os.path.join(image_dir_temp,label)
                    for image in os.listdir(label_dir):
                        if image[-3:] in ['jpg','bmp','png']:
                            image_path = os.path.join(label_dir,image)
                            self.image_full_arr.append(image_path)
                            self.label_full_arr.append(label)
                        
        else:
            label_list = os.listdir(image_dir)
            
            
            for label in label_list:
                label_dir = os.path.join(image_dir,label)
                for image in os.listdir(label_dir):
                    if image[-3:] in ['jpg','bmp','png']:
                        image_path = os.path.join(label_dir,image)
                        self.image_full_arr.append(image_path)
                        self.label_full_arr.append(label)
        
        
        if self.test_train:
            ## join both list for filtering
            arr_joined = np.column_stack((self.image_full_arr,self.label_full_arr))
            # if test_train exists, set the filter for test labels
            if self.dataset == 'pandora':
                filter = np.asarray(['10.0','14.0','16.0','20.0'])
            elif self.dataset == 'IIITD':
                filter = np.asarray(['10.0','14.0','16.0','20.0'])
            elif self.dataset == 'curtinfaces':
                filter = np.asarray(['10','14','16','20'])
            elif self.dataset == 'eurecom':
                filter = np.asarray(['10.0','14.0','16.0','20.0'])
            
#            print(arr_joined[:1])
            test_arr = arr_joined[np.in1d(arr_joined[:, 1], filter)]
            train_arr = arr_joined[np.in1d(arr_joined[:, 1], filter, invert=True)]
            
            self.image_train_arr = train_arr[:,0]
            label_train_arr = train_arr[:,1].astype(np.float)
            self.image_test_arr = test_arr[:,0]
            label_test_arr = test_arr[:,1].astype(np.float)
            
            # change label to categorical
            self.label_train_arr_onehot = to_categorical(label_train_arr)
            #find ids which are zero for all labels
            idx_train = np.argwhere(np.all(self.label_train_arr_onehot[..., :] == 0, axis=0))
            #remove id with zero labels
            self.label_train_arr_onehot = np.delete(self.label_train_arr_onehot, idx_train, axis=1)
            
            #same operation for test data
            self.label_test_arr_onehot = to_categorical(label_test_arr)
            idx_test = np.argwhere(np.all(self.label_test_arr_onehot[..., :] == 0, axis=0))
            self.label_test_arr_onehot = np.delete(self.label_test_arr_onehot, idx_test, axis=1)            

#            self.steps_per_epochs = len(image_full_arr)/batch_size
            self.num_classes_train = len(set(label_train_arr))
            self.num_classes_test = len(set(label_test_arr))  
            if test_train=='train':
                print('{} Images found in {} classes'.format(len(self.image_train_arr),self.num_classes_train))
            elif test_train=='test':
                print('{} Images found in {} classes'.format(len(self.image_test_arr),self.num_classes_test))
        else:
            self.label_full_arr_onehot = to_categorical(self.label_full_arr)
            idx_full = np.argwhere(np.all(self.label_full_arr_onehot[..., :] == 0, axis=0))
            self.label_full_arr_onehot = np.delete(self.label_test_arr_onehot, idx_full, axis=1) 

            self.num_classes_full = len(set(self.label_full_arr))
            
   
            
    def gen_batch(self, X1, X2, batch_size):
    
        while True:
            idx = np.random.choice(X1.shape[0], batch_size, replace=False)
            yield X1[idx], X2[idx]            
            
    def normalization(self, X):
        result = X / 127.5 - 1
        
        # Deal with the case where float multiplication gives an out of range result (eg 1.000001)
        out_of_bounds_high = (result > 1.)
        out_of_bounds_low = (result < -1.)
        out_of_bounds = out_of_bounds_high + out_of_bounds_low
        
        if not all(np.isclose(result[out_of_bounds_high],1)):
            raise RuntimeError("Normalization gave a value greater than 1")
        else:
            result[out_of_bounds_high] = 1.
            
        if not all(np.isclose(result[out_of_bounds_low],-1)):
            raise RuntimeError("Normalization gave a value lower than -1")
        else:
            result[out_of_bounds_low] = 1.
        
        return result       
            
            
    def flow_from_dir(self, batch_size, target_dim):
        
        
        if self.test_train == 'train':
            image_iter_arr = self.image_train_arr
            label_iter_arr = self.label_train_arr_onehot
        elif self.test_train == 'test':
            image_iter_arr = self.image_test_arr
            label_iter_arr = self.label_test_arr_onehot
        else:
            image_iter_arr = self.image_full_arr
            label_iter_arr = self.label_full_arr_onehot

        # iterate over the list of image and label batch
        for batch in self.gen_batch(image_iter_arr, label_iter_arr, batch_size):
                
            image_arr_batch = []
            image_depth_arr_batch = []
            try:
                image_path_batch = batch[0]
                #load label and convert to one hot
                y_batch = batch[1]
                
                #read and resize image to target_dim
                for image_path in image_path_batch:
                    
        #            print(image_depth_path)
                    if self.dataset == 'eurecom':
                        image_depth_path = image_path.replace('/RGB/','/depth/')
                        image_depth_path = image_depth_path.replace('rgb_','depth_')  
        #                print(image_depth_path)
                    elif self.dataset == 'pandora':
                        image_depth_path = image_path.replace('_RGB/','_depth/')
                        image_depth_path = image_depth_path.replace('_rgb.','_depth.')
                    else:
                        image_depth_path = image_path.replace('/RGB/','/depth/')
                    
                    
                    #read rgb and depth image and add to batch
                    image_arr = cv2.imread(image_path)
                    image_arr = cv2.resize(image_arr,target_dim[:-1])
                    image_arr_batch.append(image_arr)
                    #read depth images
                    image_depth_arr = cv2.imread(image_depth_path,0)
                    image_depth_arr = cv2.resize(image_depth_arr,target_dim[:-1])
                    image_depth_arr_batch.append(image_depth_arr)     

            except:
                continue          
    
    #                    
            x_batch_norm = self.normalization(np.asarray(image_arr_batch))
            x_batch_depth_norm = self.normalization(np.asarray(image_depth_arr_batch))
            x_batch_depth_norm = np.expand_dims(x_batch_depth_norm, axis=-1)
            yield x_batch_norm, x_batch_depth_norm, y_batch
    

#def class_generator_init(image_dir, batch_size, dataset, test_train=None):
##    def _init_(self, image_dir, batch_size):
#    
#    image_full_arr = []
#    label_full_arr = []
#    if isinstance(image_dir, list):
#        for image_dir_temp in image_dir:
#            label_list = os.listdir(image_dir_temp)
#            
#            for label in label_list:
#                label_dir = os.path.join(image_dir_temp,label)
#                for image in os.listdir(label_dir):
#                    image_path = os.path.join(label_dir,image)
#                    image_full_arr.append(image_path)
#                    label_full_arr.append(label)
#                    
#    else:
#        label_list = os.listdir(image_dir)
#        
#        
#        for label in label_list:
#            label_dir = os.path.join(image_dir,label)
#            for image in os.listdir(label_dir):
#                if image[-3:] in ['jpg','bmp','png']:
#                    image_path = os.path.join(label_dir,image)
#                    image_full_arr.append(image_path)
#                    label_full_arr.append(label)
#    steps_per_epochs = len(image_full_arr)/batch_size
##    num_classes = len(set(label_full_arr))
#    label_full_arr = np.asarray(label_full_arr, dtype=int)
#    
#    if dataset == 'IIITD':
#        
#        if test_train:
#
#            arr_joined = np.column_stack((image_full_arr,label_full_arr))
#            filter = np.asarray(['10.0','14.0','16.0','20.0'])
#            
#            test_arr = arr_joined[np.in1d(arr_joined[:, 1], filter)]
#            train_arr = arr_joined[np.in1d(arr_joined[:, 1], filter, invert=True)]
#            
#            image_train_arr = train_arr[:,0]
#            label_train_arr = train_arr[:,1].astype(np.float)
#            image_test_arr = test_arr[:,0]
#            label_test_arr = test_arr[:,1].astype(np.float)
#            
#            # change label to categorical
#            label_train_arr_onehot = to_categorical(label_train_arr)
#            #find ids which are zero for all labels
#            idx_train = np.argwhere(np.all(label_train_arr_onehot[..., :] == 0, axis=0))
#            #remove id with zero labels
#            label_train_arr_onehot = np.delete(label_train_arr_onehot, idx_train, axis=1)
#            
#            #same operation for test data
#            label_test_arr_onehot = to_categorical(label_test_arr)
#            idx_test = np.argwhere(np.all(label_test_arr_onehot[..., :] == 0, axis=0))
#            label_test_arr_onehot = np.delete(label_test_arr_onehot, idx_test, axis=1)            
#
#
#            num_classes_train = len(set(label_train_arr))
#            num_classes_test = len(set(label_test_arr))
#        else:
#            label_full_arr_onehot = to_categorical(label_full_arr)
#            label_full_arr_onehot = label_full_arr_onehot[:,1:]
#            num_classes = len(set(label_full_arr))
#
#    elif dataset == 'pandora':
#
#        label_full_arr = np.asarray(label_full_arr, dtype=int)
#        label_full_arr = np.ceil(label_full_arr/5)
#        num_classes = len(set(label_full_arr))
#        if test_train:
##            image_full_arr = np.expand_dims(image_full_arr,axis=1)
##            label_full_arr = np.expand_dims(label_full_arr,axis=0)
#            arr_joined = np.column_stack((image_full_arr,label_full_arr))
#            filter = np.asarray(['10.0','14.0','16.0','20.0'])
#            test_arr = arr_joined[np.in1d(arr_joined[:, 1], filter)]
##            test_arr = arr_joined[arr_joined[:, 1] in filter]
#            train_arr = arr_joined[np.in1d(arr_joined[:, 1], filter, invert=True)]
#            image_train_arr = train_arr[:,0]
#            label_train_arr = train_arr[:,1].astype(np.float)
#            image_test_arr = test_arr[:,0]
#            label_test_arr = test_arr[:,1].astype(np.float)
#            # change label to categorical
#            label_train_arr_onehot = to_categorical(label_train_arr)
#            #find ids which are zero for all labels
#            idx_train = np.argwhere(np.all(label_train_arr_onehot[..., :] == 0, axis=0))
#            #remove id with zero labels
#            label_train_arr_onehot = np.delete(label_train_arr_onehot, idx_train, axis=1)
#            
#            label_test_arr_onehot = to_categorical(label_test_arr)
#            idx_test = np.argwhere(np.all(label_test_arr_onehot[..., :] == 0, axis=0))
#            label_test_arr_onehot = np.delete(label_test_arr_onehot, idx_test, axis=1)            
##            label_test_arr_onehot = label_test_arr_onehot[:,1:]
#            num_classes_train = len(set(label_train_arr))
#            num_classes_test = len(set(label_test_arr))
#        else:
#            label_full_arr_onehot = to_categorical(label_full_arr)
#            label_full_arr_onehot = label_full_arr_onehot[:,1:]
#    else:
#        num_classes = len(set(label_full_arr))
#
#        label_full_arr_onehot = to_categorical(label_full_arr)
#        label_full_arr_onehot = label_full_arr_onehot[:,1:]
#    
#    if test_train=='train':
#        print('{} Images found in {} classes'.format(len(image_train_arr),num_classes_train))
#        return np.asarray(image_train_arr), label_train_arr_onehot, np.asarray(image_test_arr), label_test_arr_onehot, steps_per_epochs, num_classes
#    elif test_train=='test':
#        print('{} Images found in {} classes'.format(len(image_test_arr),num_classes_test))
#        return np.asarray(image_train_arr), label_train_arr_onehot, np.asarray(image_test_arr), label_test_arr_onehot, steps_per_epochs, num_classes
#    else:
#        return np.asarray(image_full_arr), label_full_arr_onehot, steps_per_epochs, num_classes
##        
#def flow_from_dir(image_dir, batch_size, target_dim):
#    
##    flip_counter = 0
#    #get the image path and label dict
#    
#    image_full_arr, label_full_arr, steps_per_epochs, num_classes = class_generator_init(image_dir, batch_size)
#    
##    random.seed(seed)
#    for batch in gen_batch_cat(image_full_arr,label_full_arr, batch_size):
#            
#        image_arr_batch = []
#        image_path_batch = batch[0]
#        #load label and convert to one hot
#        y_batch = batch[1]
#        
#        #read and resize image to target_dim
#        for image_path in image_path_batch:
#            try:
#                image_arr = cv2.imread(image_path)
#                image_arr = cv2.resize(image_arr,(256,256))
#                image_arr_batch.append(image_arr)
#            except:
#                continue
##                    
#        x_batch_norm = normalization(np.asarray(image_arr_batch))
#
#        yield x_batch_norm, y_batch
#
#def flow_from_both_dir(image_rgb_dir, batch_size, target_dim, dataset, test_train=None):
#    
##    flip_counter = 0
#    #get the image path and label dict
#    if test_train:
#        image_train_arr, label_train_arr, image_test_arr, label_test_arr, steps_per_epochs, num_classes = class_generator_init(image_rgb_dir, batch_size, dataset, test_train)
#    else:
#        image_full_arr, label_full_arr, steps_per_epochs, num_classes = class_generator_init(image_rgb_dir, batch_size, dataset)
##    random.seed(seed)
#    if test_train == 'train':
#        image_iter_arr = image_train_arr
#        label_iter_arr = label_train_arr
#    elif test_train == 'test':
#        image_iter_arr = image_test_arr
#        label_iter_arr = label_test_arr
#    else:
#        image_iter_arr = image_full_arr
#        label_iter_arr = label_full_arr
#    for batch in gen_batch_cat(image_iter_arr,label_iter_arr, batch_size):
#            
#        image_arr_batch = []
#        image_depth_arr_batch = []
#        try:
#            image_path_batch = batch[0]
#            #load label and convert to one hot
#            y_batch = batch[1]
#            
#            #read and resize image to target_dim
#            for image_path in image_path_batch:
#                image_depth_path = image_path.replace('/RGB/','/depth/')
#    #            print(image_depth_path)
#                if dataset == 'eurecom':
#                    image_depth_path = image_path.replace('/RGB/','/depth/')
#                    image_depth_path = image_depth_path.replace('rgb_','depth_')  
#    #                print(image_depth_path)
#                if dataset == 'pandora':
#                    image_depth_path = image_path.replace('_RGB/','_depth/')
#                    image_depth_path = image_depth_path.replace('_rgb.','_depth.')
#                    image_arr = cv2.imread(image_path)
#                    image_arr = cv2.resize(image_arr,target_dim[:-1])
#                    image_arr_batch.append(image_arr)
#                    #read depth images
#                    image_depth_arr = cv2.imread(image_depth_path,0)
#                    image_depth_arr = cv2.resize(image_depth_arr,target_dim[:-1])
#                    image_depth_arr_batch.append(image_depth_arr)            
#        except:
#            continue         
#
##                    
#        x_batch_norm = normalization(np.asarray(image_arr_batch))
#        x_batch_depth_norm = normalization(np.asarray(image_depth_arr_batch))
#        x_batch_depth_norm = np.expand_dims(x_batch_depth_norm, axis=-1)
#        yield x_batch_norm, x_batch_depth_norm, y_batch
#        
#
#def flow_from_both_dir_disc(image_rgb_dir, batch_size, target_dim, dataset, test_train=None):
#    
##    flip_counter = 0
#    #get the image path and label dict
#    if test_train:
#        image_train_arr, label_train_arr, image_test_arr, label_test_arr, steps_per_epochs, num_classes = class_generator_init(image_rgb_dir, batch_size, dataset, test_train)
#    else:
#        image_full_arr, label_full_arr, steps_per_epochs, num_classes = class_generator_init(image_rgb_dir, batch_size, dataset)
##    random.seed(seed)
#    if test_train == 'train':
#        image_iter_arr = image_train_arr
#        label_iter_arr = label_train_arr
#    elif test_train == 'test':
#        image_iter_arr = image_test_arr
#        label_iter_arr = label_test_arr
#    else:
#        image_iter_arr = image_full_arr
#        label_iter_arr = label_full_arr
##    print(len(image_iter_arr))
#    for batch in gen_batch(image_iter_arr, label_iter_arr, batch_size):
#            
#        image_arr_batch = []
#        image_depth_arr_batch = []
#        try:
#            image_path_batch = batch[0]
#            #load label and convert to one hot
#            y_batch = batch[1]
#            
#            #read and resize image to target_dim
#            for image_path in image_path_batch:
#                image_depth_path = image_path.replace('/RGB/','/depth/')
#    #            print(image_depth_path)
#                if dataset == 'eurecom':
#                    image_depth_path = image_path.replace('/RGB/','/depth/')
#                    image_depth_path = image_depth_path.replace('rgb_','depth_')  
#    #                print(image_depth_path)
#                if dataset == 'pandora':
#                    image_depth_path = image_path.replace('_RGB/','_depth/')
#                    image_depth_path = image_depth_path.replace('_rgb.','_depth.')
#                    image_arr = cv2.imread(image_path)
#                    image_arr = cv2.resize(image_arr,target_dim[:-1])
#                    image_arr_batch.append(image_arr)
#                    #read depth images
#                    image_depth_arr = cv2.imread(image_depth_path,0)
#                    image_depth_arr = cv2.resize(image_depth_arr,target_dim[:-1])
#                    image_depth_arr_batch.append(image_depth_arr)            
#        except:
#            continue          
#
##                    
#        x_batch_norm = normalization(np.asarray(image_arr_batch))
#        x_batch_depth_norm = normalization(np.asarray(image_depth_arr_batch))
#        x_batch_depth_norm = np.expand_dims(x_batch_depth_norm, axis=-1)
#        yield x_batch_norm, x_batch_depth_norm, y_batch


#def get_vgg_activation(model, tensor, layer_name):
##    input_tensor = Input(shape=(128,128,1))
#    pred_depth = model.predict(X_depth_batch)
##    model_edit = Model(inputs=[input_tensor], outputs=[pred_depth])
##    model = vgg16.VGG16(input_tensor=input_tensor, input_shape=(256, 256, 3), weights='imagenet', include_top=False)
#    outputs_dict = {}
#    for layer in model.layers:
##        print(layer.name)
#        outputs_dict[layer.name] = layer.output
#        layer.trainable = False
#    return outputs_dict[layer_name]


def discriminator_model_loader(image_generator, generator_model, patch_size, image_data_format, label_smoothing, label_flipping):
    batch_counter = 0
    while True:
        X_rgb_batch, X_depth_batch, y_label = next(image_generator)
        
        X_disc, y_disc = get_disc_batch(X_depth_batch, X_rgb_batch, generator_model, batch_counter, patch_size, image_data_format, label_smoothing=label_smoothing, label_flipping=label_flipping)
        batch_counter += 1
        yield X_disc, y_disc

def generator_train_loader(image_generator, classifier_model, style_layers):   
    while True:
        X_rgb_batch, X_depth_batch, y_label = next(image_generator)
#        gray_X_rgb_batch = []
#        for j in range(0,X_rgb_batch.shape[0]):
#            gray_X_rgb_batch.append(cv2.cvtColor(np.array(X_rgb_batch[j]), cv2.COLOR_BGR2GRAY))
#        gray_X_rgb_batch = np.array(gray_X_rgb_batch, dtype='float32')
#        print(gray_X_rgb_batch.shape)
#        batch_size = y_label.shape[0]
#        content_tensor = K.variable(X_depth_batch)
            
#        style_tensor = K.variable(X_depth_batch)
#        print(X_depth_batch.shape)
#        activations = keract.get_activations(classifier_model, X_depth_batch, style_layers[0])
        style_act_1 = keract.get_activations(classifier_model, X_depth_batch, style_layers[0])
        for value in style_act_1.values():
            style_act_1 = value
        style_act_2 = keract.get_activations(classifier_model, X_depth_batch, style_layers[1])
        for value in style_act_2.values():
            style_act_2 = value        
        style_act_3 = keract.get_activations(classifier_model, X_depth_batch, style_layers[2])
        for value in style_act_3.values():
            style_act_3 = value
        style_act_4 = keract.get_activations(classifier_model, X_depth_batch, style_layers[3])
        for value in style_act_4.values():
            style_act_4 = value
#        content_activation = style_act_2
        dummy_input = np.zeros_like(y_label)
        
#        yield [X_rgb_batch, content_activation, style_act_1, style_act_2, style_act_3, style_act_4], [dummy_input, dummy_input, dummy_input, dummy_input, dummy_input, dummy_input, X_depth_batch]
        yield [X_rgb_batch, style_act_1, style_act_2, style_act_3, style_act_4], [dummy_input, dummy_input, dummy_input, dummy_input, dummy_input, X_depth_batch]
        
        
        
def DCGAN_model_loader(image_generator):
    while True:
        X_rgb_batch, X_depth_batch, y_label = next(image_generator)   
        y_gen = np.ones((X_rgb_batch.shape[0], 1), dtype=np.uint8)
#        y_gen[:, 1] = 1
        
        yield X_rgb_batch, y_gen
    
    
#image_full_arr, label_full_arr, steps_per_epochs, num_classes = class_generator_init('D:/CurtinFaces_crop/RGB/train/',10)
#for x, y in flow_from_dir('D:/CurtinFaces_crop/RGB/train/',10, 256):
#    print(x,y)
#    break

def plot_generated_batch_wgan(X_full, X_sketch, generator_model, batch_size, image_data_format, suffix, logging_dir,noise_dim,noise_scale=0.5):

    # Generate images
#    X_gen = generator_model.predict(X_sketch)
    # Pass noise to the generator
#    noise_input = sample_noise(noise_scale, batch_size, noise_dim)
    # Produce an output
    X_gen,pred_class = generator_model.predict([X_sketch])
    X_sketch = inverse_normalization(X_sketch)
    X_full = inverse_normalization(X_full)
    X_gen = inverse_normalization(X_gen) 
    
    Xs = X_sketch[:8]
    Xg = X_gen[:8]
    Xr = X_full[:8]
    
    if image_data_format == "channels_last":
#        X = np.concatenate((Xs, Xg, Xr), axis=0)
        
        list_rows_rgb = []
        list_rows_depth = []
        #for RGB images
        for i in range(int(Xs.shape[0] // 4)):
            X_rgb_temp = np.concatenate([Xs[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows_rgb.append(X_rgb_temp)
        #for Depth images
        X_gen_org = np.concatenate((Xg, Xr), axis=0)
        for i in range(int(X_gen_org.shape[0] // 4)):
            
            X_depth_temp = np.concatenate([X_gen_org[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows_depth.append(X_depth_temp)

        X_rgb = np.concatenate(list_rows_rgb, axis=0)
        X_depth = np.concatenate(list_rows_depth, axis=0)

#    if image_data_format == "channels_first":
#        X = np.concatenate((Xs, Xg, Xr), axis=0)
#        list_rows = []
#        for i in range(int(X.shape[0] // 4)):
#            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
#            list_rows.append(Xr)
#
#        Xr = np.concatenate(list_rows, axis=1)
#        Xr = Xr.transpose(1,2,0)
   
    if X_rgb.shape[-1] == 1:
        plt.imshow(X_rgb[:, :, 0], cmap="gray")
    else:
        plt.imshow(X_rgb)


    plt.axis("off")
#    plt.savefig('rgb.png')
    plt.savefig(os.path.join(logging_dir, "figures/current_batch_rgb_%s.png" % suffix))
    plt.clf()
    plt.close()
    
    if X_depth.shape[-1] == 1:
        plt.imshow(X_depth[:, :, 0], cmap="gray")
    else:
        plt.imshow(X_depth)
    
    plt.axis("off")
#    plt.savefig('depth.png')
    plt.savefig(os.path.join(logging_dir, "figures/current_batch_depth_%s.png" % suffix))
    plt.clf()
    plt.close()

def plot_generated_batch(X_full, X_sketch, generator_model, batch_size, image_data_format, suffix, logging_dir):

    # Generate images
    X_gen = generator_model.predict(X_sketch)
    # Pass noise to the generator
#    noise_input = sample_noise(noise_scale, batch_size, noise_dim)
    # Produce an output
#    X_disc_gen, pred_class = generator_model.predict([noise_input,X_sketch])
    X_sketch = inverse_normalization(X_sketch)
    X_full = inverse_normalization(X_full)
    X_gen = inverse_normalization(X_gen) 
    
    Xs = X_sketch[:8]
    Xg = X_gen[:8]
    Xr = X_full[:8]
    
    if image_data_format == "channels_last":
#        X = np.concatenate((Xs, Xg, Xr), axis=0)
        
        list_rows_rgb = []
        list_rows_depth = []
        #for RGB images
        for i in range(int(Xs.shape[0] // 4)):
            X_rgb_temp = np.concatenate([Xs[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows_rgb.append(X_rgb_temp)
        #for Depth images
        X_gen_org = np.concatenate((Xg, Xr), axis=0)
        for i in range(int(X_gen_org.shape[0] // 4)):
            
            X_depth_temp = np.concatenate([X_gen_org[k] for k in range(4 * i, 4 * (i + 1))], axis=1)
            list_rows_depth.append(X_depth_temp)

        X_rgb = np.concatenate(list_rows_rgb, axis=0)
        X_depth = np.concatenate(list_rows_depth, axis=0)

#    if image_data_format == "channels_first":
#        X = np.concatenate((Xs, Xg, Xr), axis=0)
#        list_rows = []
#        for i in range(int(X.shape[0] // 4)):
#            Xr = np.concatenate([X[k] for k in range(4 * i, 4 * (i + 1))], axis=2)
#            list_rows.append(Xr)
#
#        Xr = np.concatenate(list_rows, axis=1)
#        Xr = Xr.transpose(1,2,0)
   
    if X_rgb.shape[-1] == 1:
        plt.imshow(X_rgb[:, :, 0], cmap="gray")
    else:
        plt.imshow(X_rgb)


    plt.axis("off")
#    plt.savefig('rgb.png')
    plt.savefig(os.path.join(logging_dir, "figures/current_batch_rgb_%s.png" % suffix))
    plt.clf()
    plt.close()
    
    if X_depth.shape[-1] == 1:
        plt.imshow(X_depth[:, :, 0], cmap="gray")
    else:
        plt.imshow(X_depth)
    
    plt.axis("off")
#    plt.savefig('depth.png')
    plt.savefig(os.path.join(logging_dir, "figures/current_batch_depth_%s.png" % suffix))
    plt.clf()
    plt.close()