# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:52:21 2020

@author: hardi
"""

import h5py
import os
import sys
import time
import numpy as np
import models
import keras
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras import callbacks
import tensorflow as tf
import models_WGAN
from keras.preprocessing.image import ImageDataGenerator
#import keras
import imgaug.augmenters as iaa
# Utils
sys.path.append("../utils")
import general_utils
import data_utils
import fid_calculate
#import fid_calculate as fid



def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    nb_epoch_classes = kwargs["nb_epoch_classes"]
    model_name = kwargs["model_name"]
    generator = kwargs["generator"]
    image_data_format = kwargs["image_data_format"]
    img_dim = kwargs["img_dim"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]
    do_plot = kwargs["do_plot"]
    logging_dir = kwargs["logging_dir"]
    save_every_epoch = kwargs["epoch"]
    load_weight_epoch = kwargs["load_weight_epoch"]
    batch_size_classes = kwargs["batch_size_classes"]
    data_aug = 0
    lr_G=5E-4
    lr_D=5E-4
    clamp_lower = 0.01#kwargs["clamp_lower"]
    clamp_upper = 0.01#kwargs["clamp_upper"]   
    batch_size = 30
    

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name, logging_dir=logging_dir)
    # Create optimizers
#    opt_G = data_utils.get_optimizer('RMSprop', lr_G)
#    opt_D = data_utils.get_optimizer('RMSprop', lr_D)
    opt_vgg= Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Load and rescale data
#    X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
#    img_dim = X_full_train.shape[-3:]
    img_dim = (128,128,3)
    img_dim_depth = (128,128,1)
    noise_dim = (1, 1, 128)
    n_class = 48
    noise_scale= 0.5
#    weight_file = 'D:/DeepLearningImplementations/pix2pix/ae_test1/models/CNN/gen_weights_epoch600.h5'    
    
#    # Get the number of non overlapping patch and the size of input image to the discriminator
#    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)




    # Load generator model for creating depth 
#    global generator_model

#    generator_model = models.generator_unet_upsampling_2out(img_dim, bn_mode, n_class, model_name="generator_unet_upsampling_2_out")
#    generator_model.summary()
#    generator_model.load_weights(weight_file)
#    print('weight loaded')
    
    classifier_model = models.vgg_encoder(img_dim_depth, n_class)
    
    classifier_model.summary()
#    classfier_train = models.make_classifier(classifier_model, img_dim_depth)
    
    classifier_model.compile(optimizer=opt_vgg,
                      loss=['categorical_crossentropy'],
                      metrics=['accuracy'])

    # Create the TensorBoard callback,
    # which we will drive manually
#    
#    tensorboard_train = keras.callbacks.TensorBoard(
#      log_dir=logging_dir,
#      histogram_freq=0,
#      batch_size=batch_size,
#      write_graph=True,
#      write_grads=True
#    )
#    tensorboard.set_model(classifier_model)
        
    # Transform train_on_batch return value
    # to dict expected by on_batch_end callback
#    def named_logs(model, logs):
#      result = {}
#      for l in zip(model.metrics_names, logs):
#        result[l[0]] = l[1]
#      return result
   
    
    #define data generator path
#    rgb_train_dir = 'D:/EURECOM_Kinect_Face_Dataset_crop/RGB'
#    depth_train_dir = 'D:/EURECOM_Kinect_Face_Dataset_crop/depth'
#    rgb_val_dir = 'D:/CurtinFaces_crop/DEPTH/test/'
#    depth_test_dir = 'D:/eurecom_protocol/depth/test'
    
#    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.5)
#    generator_train = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(256, 256), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42,subset='training')
#    generator_test = train_datagen.flow_from_directory(directory=rgb_val_dir, target_size=(256, 256), color_mode="rgb",
#                                                      batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42,subset='validation')

    
    rgb_train_dir1 = 'D:/CurtinFaces_crop/RGB/train'
    rgb_train_dir2 = 'D:/CurtinFaces_crop/RGB/test'
    dataset = 'curtinfaces'
    
    image_dir = [rgb_train_dir1,rgb_train_dir2]
#    generator_both_train = data_utils.flow_from_both_dir(rgb_train_dir, batch_size, img_dim,'pandora','train')
    data_gen_train = data_utils.data_pipeline(image_dir, dataset, test_train='train')
    data_gen_test = data_utils.data_pipeline(image_dir, dataset, test_train='test')  
    
    
    samples_train = 4656
    samples_test = 388
    n_batch_per_epoch = int(samples_train/(batch_size))
    epoch_size = n_batch_per_epoch * batch_size
#    print(n_batch_per_epoch)
#    image_dir_list = [rgb_train_dir, rgb_val_dir]    
#    generator_both = data_utils.flow_from_both_dir(image_dir_list, batch_size, img_dim)
#    generator_both_disc = data_utils.flow_from_both_dir_disc(image_dir_list, batch_size, img_dim)
#    generator_both_test =  data_utils.flow_from_both_dir(rgb_test_dir, batch_size, img_dim) 
    #################
    # Start training
    ################
    prev_val_loss=1000
    patience_count = 0
    max_patience = 5
    print('\n Training...')
    for e in range(nb_epoch):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        start = time.time()
#        generator_train = data_utils.flow_from_dir(depth_train_dir, batch_size, img_dim)
#        generator_test = data_utils.flow_from_dir(depth_test_dir, batch_size, img_dim)

        
        for X_rgb_batch, X_depth_batch, y_label in data_gen_train.flow_from_dir(batch_size, img_dim_depth):


            #######################
            # 2) Train the classifier
            #######################

#            X_depth_generated, pred_class = generator_model.predict([X_batch])


            classifier_logs = classifier_model.train_on_batch([X_depth_batch], [y_label])
#            tensorboard.on_epoch_end(batch_counter, named_logs(classifier_model, classifier_logs))

            batch_counter += 1
            
            progbar.add(batch_size, values=[("Loss", classifier_logs[0]),
                                            ("accuracy", classifier_logs[1])])

            # Save images for visualization 
            if batch_counter == n_batch_per_epoch :
                
                print('\n Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
                classifier_weight_path = os.path.join(logging_dir, 'models/CNN/classifier_%s.h5' % (e))
                classifier_model.save_weights(classifier_weight_path)
#                print('\n Validating...')
#                # Get new images from validation
#                val_batch = 1
#                classifier_test_logs_list = []
##                fid_score = 0
#                n_batch_per_epoch_test = samples_test/(batch_size)
#                epoch_size_test = n_batch_per_epoch_test * batch_size
#                progbar_test = generic_utils.Progbar(epoch_size_test)
#                for x_rgb_batch, x_depth_batch, y_label in generator_both_test:
##                    X_depth_generated, pred_class = generator_model.predict([x_depth_batch])
#                    
#                    classifier_test_logs = classifier_model.test_on_batch([x_depth_batch], [y_label])
#                    classifier_test_logs_list.append(classifier_test_logs)
#                    val_batch += 1
#                    
#                    progbar_test.add(batch_size, values=[("Loss", np.mean(classifier_test_logs_list,axis=0)[0]),
#                                            ("accuracy", np.mean(classifier_test_logs_list,axis=0)[1])])
#                                    
#                    if val_batch >= n_batch_per_epoch_test:
#                        print('\n Test loss:{}, Test_acc: {}'.format(np.mean(classifier_test_logs_list,axis=0)[0], np.mean(classifier_test_logs_list,axis=0)[1]))
#                        break
#                if np.mean(classifier_test_logs_list,axis=0)[0] < prev_val_loss:
#                    prev_val_loss = np.mean(classifier_test_logs_list,axis=0)[0]
#                    prev_val_acc = np.mean(classifier_test_logs_list,axis=0)[1]
#                    patience_count = 0
#
#                else:
#                    patience_count += 1
            if batch_counter >= n_batch_per_epoch:
                break
#        if patience_count >  max_patience:
#            print('\n best test accuracy: {}'.format(prev_val_acc))
#            break

        
        
#        data_utils.save_model_weights(generator_model, discriminator_model, DCGAN_model, e)
       
 