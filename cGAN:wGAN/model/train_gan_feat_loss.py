import os
import sys
import time
import pickle
import numpy as np
import models as model_lib
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
#from keras import callbacks
import tensorflow as tf
from training_fit_models import fit_models
#import keras
from tensorflow.keras.callbacks import TensorBoard
from callbacks import DecoderSnapshot, ModelsCheckpoint
#import math
import imgaug.augmenters as iaa
# Utils
sys.path.append("../utils")
import general_utils
import data_utils



def l2_norm(y_true, y_pred):
    return K.sqrt(K.sum(K.sum(K.square(y_pred - y_true), axis=-1),axis=[-2,-1]))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.mean(K.abs(y_pred - y_true), axis=-1),axis=[-2,-1])


def mean_squared_error(y_true, y_pred):
    return K.mean(K.mean(K.square(y_pred - y_true), axis=-1),axis=[-2,-1])

def sum_absolute_error(y_true, y_pred):
    return K.sum(K.sum(K.abs(y_pred - y_true), axis=-1),axis=[-2,-1])


def sum_squared_error(y_true, y_pred):
    return K.sum(K.sum(K.square(y_pred - y_true), axis=-1),axis=[-2,-1])

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.mean(K.square(y_pred - y_true), axis=-1),axis=[-2,-1]))

def rmsle(y_true, y_pred):
	return K.sqrt(K.mean(K.mean(K.square(y_pred - y_true), axis=-1),axis=[-2,-1]))



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
#    full_training = 'gan'
#    data_aug = 0
#    classifier_epochs = 100
    
#    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name, logging_dir=logging_dir)
    
    # Load and rescale data
#    X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
#    img_dim = X_full_train.shape[-3:]
    img_dim = (128,128,3)
    img_dim_depth = (128,128,1)
    n_class = 48
    classifier_weight_path = 'D:/DeepLearningImplementations/pix2pix/curtin_feat_loss/models/CNN/classifier_18.h5'
#    content_layers = ['conv2_1']
    style_layers = ['conv1_1', 'conv2_1',
                  'conv3_1','conv4_1']    

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim_depth, patch_size, image_data_format)


    try:
#        sess = tf.Session()
        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        opt_gen_train = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
#        global generator_model
        generator_model = model_lib.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size,
                                      do_plot)
        
        generator_model_rev = model_lib.generator_unet_upsampling_rev(img_dim_depth, bn_mode, model_name="generator_unet_upsampling_rev")
        # Load discriminator model
        discriminator_model = model_lib.load("DCGAN_discriminator",
                                          img_dim_disc,
                                          nb_patch,
                                          bn_mode,
                                          use_mbd,
                                          batch_size,
                                          do_plot)

        generator_model.compile(loss='mae', optimizer=opt_discriminator)
       
        #load classifier
        opt_vgg= Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        classifier_model = model_lib.vgg_encoder(img_dim_depth, n_class)
        classifier_model.summary()

        if classifier_weight_path:
            classifier_model.load_weights(classifier_weight_path)
            classifier_model.compile(optimizer=opt_vgg,
              loss=['categorical_crossentropy'],
              metrics=['accuracy'])         
        
        
        
        
        discriminator_model.trainable = False
        classifier_model.trainable = False
        
        generator_train = model_lib.build_graph_perceptual_loss(generator_model, classifier_model, img_dim)
        DCGAN_model = model_lib.DCGAN(generator_model, discriminator_model, img_dim, patch_size, image_data_format)
        
        
#        loss = ['categorical_crossentropy']
#        loss_weights = [1E1, 1]
#        generator_train.compile(loss=['categorical_crossentropy'], optimizer=opt_dcgan, metrics=['accuracy'])
        DCGAN_model.compile(loss=['mse'], optimizer=opt_dcgan, metrics=['accuracy'])
        
        discriminator_model.trainable = True
#        classifier_model.trainable = True
        generator_train.compile(
                loss={
#                        'content': model_lib.dummy_loss_function, 
                      'style1': model_lib.dummy_loss_function, 'style2': model_lib.dummy_loss_function,
                    'style3': model_lib.dummy_loss_function, 'style4': model_lib.dummy_loss_function, 'tv': model_lib.dummy_loss_function,
                    'generator_unet_upsampling': 'mse'},
                      optimizer=opt_gen_train,
                      loss_weights={
#                              'content': 0,
                              'style1': 1e-4, 'style2':1e2, 'style3': 1e-4, 'style4': 1e-4, 'tv': 1,
                    'generator_unet_upsampling': 1e-1})
        
        generator_train.summary()
        discriminator_model.compile(loss=['mse'], optimizer=opt_discriminator, metrics=['accuracy'])



       
        
        epoch_format = '.{epoch:03d}.h5'
        
        rgb_train_dir1 = 'D:/CurtinFaces_crop/RGB/train'
        rgb_train_dir2 = 'D:/CurtinFaces_crop/RGB/test'
        dataset = 'curtinfaces'
        
        image_dir = [rgb_train_dir1,rgb_train_dir2]
    #    generator_both_train = data_utils.flow_from_both_dir(rgb_train_dir, batch_size, img_dim,'pandora','train')
        data_gen_train = data_utils.data_pipeline(image_dir, dataset, test_train='train')
        data_gen_test = data_utils.data_pipeline(image_dir, dataset, test_train='test')
#        generator_both_test = data_utils.flow_from_both_dir_disc(image_dir, batch_size, img_dim,'pandora','test')
#        generator_both_train = data_utils.flow_from_both_dir_disc(image_dir, batch_size, img_dim,'pandora','train')
        
        checkpoint = ModelsCheckpoint(epoch_format, discriminator_model, generator_model, DCGAN_model)
#        checkpoint = ModelsCheckpoint(epoch_format, generator_model)
        decoder_sampler = DecoderSnapshot()        
        callbacks = [checkpoint, decoder_sampler, TensorBoard(log_dir=logging_dir)]
        
        dis_loader = data_utils.discriminator_model_loader(data_gen_train.flow_from_dir(batch_size, img_dim), generator_model, patch_size, image_data_format, label_smoothing, label_flipping)
        gen_loader = data_utils.generator_train_loader(data_gen_train.flow_from_dir(batch_size, img_dim), classifier_model, style_layers)
        gan_loader = data_utils.DCGAN_model_loader(data_gen_train.flow_from_dir(batch_size, img_dim))        
        
        models = [discriminator_model,discriminator_model, generator_train, DCGAN_model]
        generators = [dis_loader, dis_loader, gen_loader, gan_loader]
#        
        metrics = [{'di_l1': 0, 'di_acc1': 1}, {'di_l2': 0, 'di_acc2': 1}, {'gen_l': 0, 'gen_style_loss1': 1, 'gen_style_loss2': 2, 'gen_style_loss3': 3, 'gen_style_loss4': 4, 'gen_tv': 5,'gen_mse':6}, {'gan_l': 0,'gan_acc': 1}]
#        models = [generator_train]
#        generators = [gen_loader]
#        
#        metrics = [ {'gen_l': 0, 'gen_style_loss1': 1, 'gen_style_loss2': 2, 'gen_style_loss3': 3, 'gen_style_loss4': 4, 'gen_tv': 5,'gen_mse':6}]

    
        histories = fit_models(generator_model, models, generators, metrics, batch_size, validation_img_gen = data_gen_test.flow_from_dir(batch_size, img_dim),
                               steps_per_epoch=(4656*4/batch_size), callbacks=callbacks,epochs=nb_epoch)        
        with open('histories.pickle', 'wb') as f:
            pickle.dump(histories, f)

#            
    except KeyboardInterrupt:
        pass
