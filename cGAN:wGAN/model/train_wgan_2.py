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
    lr_G=5E-4
    lr_D=5E-4
    clamp_lower = 0.01#kwargs["clamp_lower"]
    clamp_upper = 0.01#kwargs["clamp_upper"]    
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name, logging_dir=logging_dir)
    # Create optimizers
    opt_G = data_utils.get_optimizer('RMSprop', lr_G)
    opt_D = data_utils.get_optimizer('RMSprop', lr_D)

    # Load and rescale data
#    X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
#    img_dim = X_full_train.shape[-3:]
    img_dim = (256,256,3)
    noise_dim = (1, 1, 128)
    n_class = 52
    noise_scale= 0.5
    
    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)
    



    # Load model
    global generator_model
    
    generator_model = models.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size,
                                      do_plot) 
    generator_model.compile(loss='mae', optimizer=opt_G)

#    generator_model = models_WGAN.generator_upsampling(noise_dim, img_dim, bn_mode,n_class, model_name="generator_upsampling")

    discriminator_model = models_WGAN.discriminator(img_dim_disc, bn_mode, nb_patch, use_mbd)
    DCGAN_model = models_WGAN.DCGAN_wgan(generator_model, discriminator_model, noise_dim, img_dim, patch_size, image_data_format)
#    DCGAN_model = models_WGAN.DCGAN(generator_model, discriminator_model, noise_dim, img_dim)        

    ############################
    # Compile models
    ############################
#    loss_weights = [0.5, 0.5]
#    generator_model.compile(loss=['mae','categorical_crossentropy'], optimizer=opt_G, metrics=['accuracy'])
#    loss_weights_gan = [1, 1E1]
    discriminator_model.trainable = False
    DCGAN_model.compile(loss=[ models_WGAN.wasserstein], optimizer=opt_G, metrics=['accuracy'])
    discriminator_model.trainable = True
    discriminator_model.compile(loss=models_WGAN.wasserstein, optimizer=opt_D)        
    
    ##load weights
    generator_model.load_weights('D:/DeepLearningImplementations/pix2pix/ae/models/CNN/gen_weights_epoch150.h5')
#    generator_model.load_weights('D:/DeepLearningImplementations/pix2pix/ae_wgan_patches_3/models/CNN/gen_weights_epoch30.h5')
    print('weight loaded')

    # Global iteration counter for generator updates
    gen_iterations = 0
#    rgb_train_dir = 'D:/CurtinFaces_crop/RGB/train/'
#    rgb_val_dir = 'D:/CurtinFaces_crop/RGB/test/'
#    rgb_test_dir = 'D:/CurtinFaces_crop/RGB/test1'
    
#    rgb_train_dir = 'D:/EURECOM_Kinect_Face_Dataset_crop/RGB/'
#    rgb_val_dir='D:/RGB_D_Dataset_new-iiitd/fold1/test/RGB/'
    
#    image_dir_list = [rgb_train_dir, rgb_val_dir]    
#    generator_both = data_utils.flow_from_both_dir(image_dir_list, batch_size, img_dim)
#    generator_both_disc = data_utils.flow_from_both_dir_disc(rgb_train_dir, batch_size, img_dim,'eurecom')
    rgb_train_dir = 'D:/Pandora/face_dataset_RGB/'
    generator_both_train = data_utils.flow_from_both_dir_disc(rgb_train_dir, batch_size, img_dim,'pandora','train')
    generator_both_test = data_utils.flow_from_both_dir_disc(rgb_train_dir, batch_size, img_dim,'pandora','train')
#    generator_both_disc_val = data_utils.flow_from_both_dir_disc(rgb_val_dir, batch_size, img_dim,'IIITD')
#    generator_both_test =  data_utils.flow_from_both_dir(rgb_test_dir, batch_size, img_dim) 
    #################
    # Start training
    ################
    for e in range(nb_epoch):
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        start = time.time()
        generator_both = data_utils.flow_from_both_dir(rgb_train_dir, batch_size, img_dim,'eurecom')

#                depth_train_dir = 'D:/CurtinFaces_crop/DEPTH/train/'
#                depth_val_dir = 'D:/CurtinFaces_crop/DEPTH/test1/'                
        
              

        for X_rgb_batch, X_depth_batch, y_label in generator_both:

            if gen_iterations < 10 or gen_iterations%500 == 0:
                disc_iterations = 200
            else:
                disc_iterations = 5

            ###################################
            # 1) Train the critic / discriminator
            ###################################
            list_disc_loss_real = []
            list_disc_loss_gen = []
#            list_ae_loss = []
#            print('training critic for {}'.format(disc_iterations))
#            progbar_disc = generic_utils.Progbar(disc_iterations)
            
            for disc_it in range(disc_iterations):

                # Clip discriminator weights
                for l in discriminator_model.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, clamp_lower, clamp_upper) for w in weights]
                    l.set_weights(weights)

                X_rgb_batch1, X_depth_batch1, y_label1 = next(generator_both_disc)

                # Create a batch to feed the discriminator model
                X_disc_real, X_disc_gen = data_utils.get_disc_batch_wgan(X_depth_batch1,X_rgb_batch1, generator_model, batch_counter, batch_size, patch_size, image_data_format, noise_dim, noise_scale=0.5)
#                print(X_disc_real[0].shape, len(X_disc_real))
                # Update the discriminator
#                label_length = X_disc_real[0].shape[0] * len(X_disc_real)
                disc_loss_real = discriminator_model.train_on_batch(X_disc_real, -np.ones(X_disc_real[0].shape[0]))
                disc_loss_gen = discriminator_model.train_on_batch(X_disc_gen, np.ones(X_disc_gen[0].shape[0]))
                list_disc_loss_real.append(disc_loss_real)
                list_disc_loss_gen.append(disc_loss_gen)
                ##train ae with disc to reduce the 
#                X_gen_noise = data_utils.sample_noise(noise_scale, batch_size, noise_dim)
#                ae_loss = generator_model.train_on_batch([X_gen_noise,X_rgb_batch], [X_depth_batch,y_label])
#                list_ae_loss.append(ae_loss)
#                progbar_disc.add(1, values=[("Loss_D", -disc_loss_real - disc_loss_gen),
#                                                ("Loss_D_real", -disc_loss_real),
#                                                ("Loss_D_gen", disc_loss_gen)
##                                                ("Loss_ae_mae", ae_loss[0]),
##                                                ("Loss_ae_cat",ae_loss[1])
#                                                ])

            #######################
            # 2) Train the generator
            #######################
#            X_gen_noise = data_utils.sample_noise(noise_scale, batch_size, noise_dim)
#            print('training generator')
#            generator_both = data_utils.flow_from_both_dir(image_dir_list, batch_size, img_dim)
#            for X_sketch_batch, X_full_batch, y_label in generator_both:
#            X_gen_noise = data_utils.sample_noise(noise_scale, batch_size, noise_dim)
        # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch([X_rgb_batch], [-np.ones(X_rgb_batch.shape[0])])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            gen_iterations += 1
            batch_counter += 1
            progbar.add(batch_size, values=[("Loss_D", -np.mean(list_disc_loss_real) - np.mean(list_disc_loss_gen)),
                                            ("Loss_D_real", -np.mean(list_disc_loss_real)),
                                            ("Loss_D_gen", np.mean(list_disc_loss_gen)),
#                                                ("Loss_ae_mae", np.mean(list_ae_loss,axis=0)[0]),
#                                                ("Loss_ae_cat", np.mean(list_ae_loss,axis=0)[1]),
#                                            ("WGAN tot", gen_loss[0]),
                                            ("WGAN disc loss", -gen_loss[0])])
#                                            ("WGAN mae", gen_loss[2]),
#                                            ("WGAN cat loss", gen_loss[3]),
#                                            ("WGAN cat acc", gen_loss[6])])

            # Save images for visualization 
            if batch_counter == n_batch_per_epoch :
                data_utils.plot_generated_batch_wgan(X_depth_batch, X_rgb_batch, generator_model,
                                                batch_size, image_data_format, "training",
                                                logging_dir,noise_dim)

#                # Get new images from validation
#                val_batch = 0
#  
#                fid_score = 0

#                        fd = fid.FrechetInceptionDistance(generator_model, (0,1)) 

                if e%10 == 0:
                    X_sketch_batch, X_full_batch, y_label = next(generator_both_disc_val)
#                    for X_sketch_batch, X_full_batch, y_label in generator_both_test:
#                        noise_input = data_utils.sample_noise(noise_scale, batch_size, noise_dim)
#                        x_batch_depth = generator_model.predict([noise_input,X_sketch_batch])
#                        
#                        try:
#                            fid_score += fid_calculate.calculate_fid(x_batch_depth,X_full_batch)
#                            val_batch+=1
#                        except:
#                            continue
#                        if val_batch == 10:
#                            break
#                    print_fid_score = fid_score/10
#
#                    
#                    print('\n FID: {}'.format(print_fid_score))
#                        X_full_batch, X_sketch_batch = next(data_utils.gen_batch(X_full_val, X_sketch_val, batch_size))
                    data_utils.plot_generated_batch_wgan(X_full_batch, X_sketch_batch, generator_model,
                                                    batch_size, image_data_format, "validation",
                                                    logging_dir,noise_dim)
                    # Save model weights (by default, every 100 epochs)
                    gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_epoch%s.h5' % (model_name, e))
                    generator_model.save_weights(gen_weights_path, overwrite=True)
                    disc_weights_path = os.path.join(logging_dir, 'models/%s/disc_weights_epoch%s.h5' % (model_name, e))
                    discriminator_model.save_weights(disc_weights_path, overwrite=True)
    
            
                    print('\nEpoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
            if batch_counter >= n_batch_per_epoch:
                break

        
#        data_utils.save_model_weights(generator_model, discriminator_model, DCGAN_model, e)
       
 