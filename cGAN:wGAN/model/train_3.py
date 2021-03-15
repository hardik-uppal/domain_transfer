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
#import keras
import imgaug.augmenters as iaa
# Utils
sys.path.append("../utils")
import general_utils
import data_utils
import fid_calculate
#import fid_calculate as fid


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


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
    full_training = 'gan'
    data_aug = 0
    
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name, logging_dir=logging_dir)

    # Load and rescale data
#    X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
#    img_dim = X_full_train.shape[-3:]
    img_dim = (256,256,3)
    

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)


    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        global generator_model
#        generator_model = models.load("generator_unet_%s_2out" % generator,
#                                      img_dim,
#                                      nb_patch,
#                                      bn_mode,
#                                      use_mbd,
#                                      batch_size,
#                                      do_plot)
        generator_model = models.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size,
                                      do_plot) 
        generator_model.compile(loss='mae', optimizer=opt_discriminator)
#        # Load discriminator model
#        discriminator_model = models.load("DCGAN_discriminator",
#                                          img_dim_disc,
#                                          nb_patch,
#                                          bn_mode,
#                                          use_mbd,
#                                          batch_size,
#                                          do_plot)
#        vgg_encoder = models.load("vgg_encoder",
#                                          img_dim,
#                                          nb_patch,
#                                          bn_mode,
#                                          use_mbd,
#                                          batch_size,
#                                          do_plot)        

#        generator_model.compile(loss=['mae','categorical_crossentropy'], optimizer=opt_discriminator, metrics=['accuracy'])
        
#        if load_weight_epoch is not None:
##            
#            generator_model.load_weights('D:/DeepLearningImplementations/pix2pix/dcgan_test1/models/CNN/gen_weights_epoch%s.h5'% (load_weight_epoch))
#            generator_model._make_predict_function
#            global graph
#            graph = tf.get_default_graph()
#            print('Generator weights loaded from previous epoch:%s '% (load_weight_epoch))
#            generator_model._make_predict_function()
#            graph = tf.get_default_graph()
        
#        
#        if load_weight_epoch is not None:
#            discriminator_model.load_weights('D:/DeepLearningImplementations/pix2pix/class_disc_add/models/CNN/disc_weights_epoch%s.h5'% (load_weight_epoch))
#            print('Discriminator weights loaded from previous epoch:%s '% (load_weight_epoch))
#
#        #encoder-genrator model
##        '''
#        discriminator_model.trainable = False
##        n_class =52
##        encoder_GAN_model = models.encoder_GAN(generator_model,
##                                   discriminator_model,
##                                   n_class,
##                                   img_dim,
##                                   patch_size,
##                                   image_data_format)        
#        
#
#
#        DCGAN_model = models.DCGAN(generator_model,
#                                   discriminator_model,
#                                   img_dim,
#                                   patch_size,
#                                   image_data_format)
#        
#        
##        loss = [l1_loss, 'binary_crossentropy', 'categorical_crossentropy']
#        loss = [l1_loss, 'binary_crossentropy']
#        loss_weights = [1E1, 1]
#        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)
##        encoder_GAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)        
#        discriminator_model.trainable = True
#        
#        
#        
#        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator, metrics=['accuracy'])
##        '''
       
        
        
        
#        depth_guided_att = models.load("depth_guided_att", img_dim, nb_patch, bn_mode, use_mbd, batch_size, do_plot)   
#        # compile the model
#
#        depth_guided_att.compile(optimizer=Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
#                      loss={'output':'categorical_crossentropy','output_rgb':'categorical_crossentropy','output_depth':'categorical_crossentropy'},#triplet_loss_adapted_from_tf,
#                      metrics=['accuracy']) 
#        depth_guided_att.load_weights('D:/tutorial/AE-Gan_reperesentation/CurtinFaces_cropped/depth_guided_att_ratio8_2_no_dropout/weights-best.h5')
#        print('depth_guided_att weights loaded')
#        
#        depth_guided_att.trainable = False
#        CATGAN_model = models.CATGAN(generator_model,
#                                   depth_guided_att,
#                                   img_dim,
#                                   image_data_format) 
#        loss = [l1_loss, 'categorical_crossentropy']
#        loss_weights = [1E1, 1]
#        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)        
#        depth_guided_att.trainable = True        
        gen_loss = 100
        disc_loss = 100

        # Start training
        if full_training in ['both','gan']:
            print("Start training")
            for e in range(nb_epoch):
                # Initialize progbar and batch counter
                progbar = generic_utils.Progbar(epoch_size)
                batch_counter = 1
                start = time.time()
                rgb_train_dir = 'D:/CurtinFaces_crop/RGB/train/'
                rgb_val_dir = 'D:/CurtinFaces_crop/RGB/test/'
                image_dir_list = [rgb_train_dir, rgb_val_dir]
#                depth_train_dir = 'D:/CurtinFaces_crop/DEPTH/train/'
#                depth_val_dir = 'D:/CurtinFaces_crop/DEPTH/test1/'                
                
                generator_both = data_utils.flow_from_both_dir(image_dir_list, batch_size, img_dim)
#                generator_both_val =  data_utils.flow_from_both_dir(rgb_val_dir, batch_size, img_dim)                
                
#                generator_depth = data_utils.flow_from_dir(depth_train_dir,batch_size,img_dim)
#                generator_depth_val =  data_utils.flow_from_dir(depth_val_dir,batch_size,img_dim)                 
    
                for X_sketch_batch, X_full_batch, y_label in generator_both:                   #data_utils.gen_batch(X_full_train, X_sketch_train, batch_size):
    
#                    # Create a batch to feed the discriminator model
##                    ''' commented for 1st part of training
#                    X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
#                                                               X_sketch_batch,
##                                                               y_label,
#                                                               generator_model,
#                                                               batch_counter,
#                                                               patch_size,
#                                                               image_data_format,
#                                                               label_smoothing=label_smoothing,
#                                                               label_flipping=label_flipping)
#    
#                    # Update the discriminator
#                    disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)
##                    '''
#    
#                    # Create a batch to feed the generator model
##                    X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
##                    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
##                    y_gen[:, 1] = 1
#                    X_gen_target, X_gen = X_full_batch, X_sketch_batch
                    # for class disc
#                    y_filler = np.zeros((X_gen.shape[0], 52), dtype=np.uint8) 
#                    y_gen = np.concatenate((y_filler, y_label), axis=1) 
#                    gen_loss = generator_model.train_on_batch(X_gen, [X_gen_target, y_label])
                    
                    ##for real or fake
#                    '''
#                    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
#                    y_gen[:, 1] = 1
                    gen_loss = generator_model.train_on_batch(X_sketch_batch, [X_full_batch])
#                    # Freeze the discriminator
#                    discriminator_model.trainable = False
#                    gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
##                    gen_loss = encoder_GAN_model.train_on_batch(X_gen, [X_gen_target, y_gen, y_label])
#                    # Unfreeze the discriminator
#                    discriminator_model.trainable = True
#                    '''
                    batch_counter += 1
#                    progbar.add(batch_size, values=[("D logloss", disc_loss[0]),
#                                                    ("D acc", disc_loss[1]),
#                                                    ("G tot", gen_loss[0]),
#                                                    ("G L1", gen_loss[1]),
#                                                    ("G logloss", gen_loss[2])])
                                                    
                    
                    progbar.add(batch_size, values=[("G tot", gen_loss)])
                    
                    #calculate FID score
                    
    
                    # Save images for visualization
                    if batch_counter == n_batch_per_epoch :
                        # Get new images from validation
                        val_batch = 0
#                        val_loss = 0
                        val_acc1 = 0 
#                        val_acc2 = 0 
#                        val_acc3 = 0   
                        fid_score = 0
                        data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                        batch_size, image_data_format, "training",
                                                        logging_dir)
#                        fd = fid.FrechetInceptionDistance(generator_model, (0,1)) 

                        if e%10 == 0:
##                        X_sketch_batch, X_full_batch, y_label = next(generator_both_val)
#                            for X_sketch_batch, X_full_batch, y_label in generator_both_val:
#    
#                                x_batch_depth = generator_model.predict(X_sketch_batch)
#                                
##                                val_acc1 += K.mean(K.equal(K.argmax(y_label, axis=-1), K.argmax(y_pred, axis=-1)))
#        #
##                                out_loss = depth_guided_att.test_on_batch([X_sketch_batch, x_batch_depth], [y_label, y_label, y_label])
##                                val_loss += out_loss[1]
##                                val_acc1 += out_loss[4]
##                                val_acc2 += out_loss[5]
##                                val_acc3 += out_loss[6]
##                                fid_score += fd(X_full_batch, X_sketch_batch)
#                                try:
#                                    fid_score += fid_calculate.calculate_fid(x_batch_depth,X_full_batch)
#                                    val_batch+=1
#                                except:
#                                    continue
#                                if val_batch == 10:
#                                    break
#                            print_fid_score = fid_score/int(10)
##                            print_val_loss = val_loss/int(2028 / int(batch_size))
##                            print_val_acc1 = val_acc1/int(2028 / int(batch_size))
##                            print_val_acc2 = val_acc2/int(2028 / int(batch_size))
##                            print_val_acc3 = val_acc3/int(2028 / int(batch_size))
##                            print(' Val loss: {} ; Val Accuracy: {}, RGB:{}, Depth:{} and FID score:{}'.format(print_val_loss, print_val_acc1, print_val_acc2, print_val_acc3,fid_score))
##                            print(' val acc{}'.format(print_val_acc1))
#                            print('FID: {}'.format(print_fid_score))
                            gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_epoch%s.h5' % (model_name, e))
                            generator_model.save_weights(gen_weights_path, overwrite=True)
    #                        X_full_batch, X_sketch_batch = next(data_utils.gen_batch(X_full_val, X_sketch_val, batch_size))
                            data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                            batch_size, image_data_format, "validation",
                                                            logging_dir)
    
                    if batch_counter >= n_batch_per_epoch:
                        break
    
                print("")
                print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
    
            
    except KeyboardInterrupt:
        pass
