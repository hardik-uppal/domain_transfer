import os
import sys
import time
import numpy as np
import models
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
        generator_model = models.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size,
                                      do_plot)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          img_dim_disc,
                                          nb_patch,
                                          bn_mode,
                                          use_mbd,
                                          batch_size,
                                          do_plot)

        generator_model.compile(loss='mae', optimizer=opt_discriminator)
        if load_weight_epoch is not None:
            
            generator_model.load_weights('D:/DeepLearningImplementations/pix2pix/class_disc_add/models/CNN/gen_weights_epoch%s.h5'% (load_weight_epoch))
            generator_model._make_predict_function
            global graph
            graph = tf.get_default_graph()
            print('Generator weights loaded from previous epoch:%s '% (load_weight_epoch))
            generator_model._make_predict_function()
            graph = tf.get_default_graph()
        
        
        if load_weight_epoch is not None:
            discriminator_model.load_weights('D:/DeepLearningImplementations/pix2pix/class_disc_add/models/CNN/disc_weights_epoch%s.h5'% (load_weight_epoch))
            print('Discriminator weights loaded from previous epoch:%s '% (load_weight_epoch))
        
        
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_size,
                                   image_data_format)
        
        
        loss = [l1_loss, 'categorical_crossentropy']
        loss_weights = [1E1, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)
        
        discriminator_model.trainable = True
        
        
        
        discriminator_model.compile(loss='categorical_crossentropy', optimizer=opt_discriminator, metrics=['accuracy'])
        if load_weight_epoch is not None:
            DCGAN_model.load_weights('D:/DeepLearningImplementations/pix2pix/class_disc_add/models/CNN/DCGAN_weights_epoch%s.h5'% (load_weight_epoch))
            print('DCGAN weights loaded from previous epoch:%s '% (load_weight_epoch))
       
        
        
        
        depth_guided_att = models.load("depth_guided_att", img_dim, nb_patch, bn_mode, use_mbd, batch_size, do_plot)   
        # compile the model
        depth_guided_att.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      loss={'output':'categorical_crossentropy'},#triplet_loss_adapted_from_tf,
                      metrics=['accuracy']) 
#        depth_guided_att.load_weights('D:/tutorial/AE-Gan_reperesentation/CurtinFaces_cropped/depth_guided_att_ratio8_2_no_dropout/weights-best.h5')
        print('depth_guided_att weights loaded')
#        
        depth_guided_att.trainable = False
        CATGAN_model = models.CATGAN(generator_model,
                                   depth_guided_att,
                                   img_dim,
                                   image_data_format) 
        loss = [l1_loss, 'categorical_crossentropy']
        loss_weights = [1E1, 1]
        CATGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08))        
        depth_guided_att.trainable = True        


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
                rgb_val_dir = 'D:/CurtinFaces_crop/RGB/test1/'

#                depth_train_dir = 'D:/CurtinFaces_crop/DEPTH/train/'
#                depth_val_dir = 'D:/CurtinFaces_crop/DEPTH/test1/'                
                
                generator_both = data_utils.flow_from_both_dir(rgb_train_dir, batch_size, img_dim)
                generator_both_val =  data_utils.flow_from_both_dir(rgb_val_dir, batch_size, img_dim)                
                
#                generator_depth = data_utils.flow_from_dir(depth_train_dir,batch_size,img_dim)
#                generator_depth_val =  data_utils.flow_from_dir(depth_val_dir,batch_size,img_dim)                 
    
                for X_sketch_batch, X_full_batch, y_label in generator_both:                   #data_utils.gen_batch(X_full_train, X_sketch_train, batch_size):
    
                    # Create a batch to feed the discriminator model
                    X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
                                                               X_sketch_batch,
#                                                               y_label,
                                                               generator_model,
                                                               batch_counter,
                                                               patch_size,
                                                               image_data_format,
                                                               label_smoothing=label_smoothing,
                                                               label_flipping=label_flipping)
    
                    # Update the discriminator
                    disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)
                    
#                    X_class, y_class = data_utils.get_batch_cnn(X_full_batch,
#                                                               X_sketch_batch,
#                                                               y_label,
#                                                               generator_model,
#                                                               batch_counter,
#                                                               patch_size,
#                                                               image_data_format,
#                                                               label_smoothing=label_smoothing,
#                                                               label_flipping=label_flipping)
                    
                    class_loss = depth_guided_att.train_on_batch([X_sketch_batch,X_full_batch], y_label)
    
                    # Create a batch to feed the generator model
#                    X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
#                    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
#                    y_gen[:, 1] = 1
                    X_gen_target, X_gen = X_full_batch, X_sketch_batch
 
#                    X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
                    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                    y_gen[:, 1] = 1                 
                    
#                    y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
#                    y_gen[:, 1] = 1
    
                    # Freeze the discriminator
                    depth_guided_att.trainable = False
                    discriminator_model.trainable = False
                    cat_loss = CATGAN_model.train_on_batch(X_gen, [X_gen_target, y_label])
                    gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                    # Unfreeze the discriminator
                    depth_guided_att.trainable = True
                    discriminator_model.trainable = True
    
                    batch_counter += 1
                    progbar.add(batch_size, values=[("D logloss", disc_loss[0]),
                                                    ("D acc", disc_loss[1]),
                                                    ("G tot", gen_loss[0]),
                                                    ("G L1", gen_loss[1]),
                                                    ("G logloss", gen_loss[2]),
                                                    ("class loss", class_loss[0]),
                                                    ("class acc", class_loss[1]),
                                                    ("CATGAN loss and acc", cat_loss[0]),
                                                    ("CATGAN loss and acc", cat_loss[1]),
                                                    ("CATGAN loss and acc", cat_loss[2])])
                    #TODO
                    #calculate FID score
                    
    
                    # Save images for visualization
                    if batch_counter == n_batch_per_epoch :
                        # Get new images from validation
                        val_loss = 0
                        val_acc1 = 0 
                        val_acc2 = 0 
                        val_acc3 = 0                         
                        data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                        batch_size, image_data_format, "training",
                                                        logging_dir)
#                        X_sketch_batch, X_full_batch, y_label = next(generator_both_val)
                        for X_sketch_batch, X_full_batch, y_label in generator_both_val:

                            x_batch_depth = generator_model.predict(X_sketch_batch)
    #
                            out_loss = depth_guided_att.test_on_batch([X_sketch_batch, x_batch_depth], [y_label])
                            val_loss += out_loss[0]
                            val_acc1 += out_loss[1]
#                            val_acc2 += out_loss[5]
#                            val_acc3 += out_loss[6]
                        
                        print_val_loss = val_loss/int(2028 / int(batch_size))
                        print_val_acc1 = val_acc1/int(2028 / int(batch_size))
#                        print_val_acc2 = val_acc2/int(2028 / int(batch_size))
#                        print_val_acc3 = val_acc3/int(2028 / int(batch_size))
                        print('; Val loss: {} , Val Accuracy: {}'.format(print_val_loss, print_val_acc1))
#                        X_full_batch, X_sketch_batch = next(data_utils.gen_batch(X_full_val, X_sketch_val, batch_size))
                        data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                        batch_size, image_data_format, "validation",
                                                        logging_dir)
    
                    if batch_counter >= n_batch_per_epoch:
                        break
    
                print("")
                print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))
    
                if e % save_every_epoch == 0:
                    gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_epoch%s.h5' % (model_name, e))
                    generator_model.save_weights(gen_weights_path, overwrite=True)
#                    generator_model.save('D:/DeepLearningImplementations/pix2pix/class_disc_add/models/CNN/gen_model.h5')
    
                    disc_weights_path = os.path.join(logging_dir, 'models/%s/disc_weights_epoch%s.h5' % (model_name, e))
                    discriminator_model.save_weights(disc_weights_path, overwrite=True)
    
                    DCGAN_weights_path = os.path.join(logging_dir, 'models/%s/DCGAN_weights_epoch%s.h5' % (model_name, e))
                    DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)
#        if full_training in ['both','classifier']:
#            prev_val_loss = 10000
#            # Start class specific training
#            print("Start class specific training")
#            for e in range(nb_epoch_classes):
##                from keras.preprocessing.image import ImageDataGenerator
##                Initialize progbar and batch counter
##                progbar = generic_utils.Progbar(epoch_size)
#                batch_counter = 1
#                start = time.time()
#                    
#                # callbacks
##                lr_class = 0.00001
##            from keras.preprocessing.image import ImageDataGenerator
#                #batch size adjusted for data aug
#                flip_counter = 0
#                if data_aug == 1:
#                    batch_size = int(batch_size_classes/3)
#                    
#                else:
#                    batch_size = batch_size_classes
#                save_dir = os.path.join(logging_dir,'depth_guided_cnn')
##                log = callbacks.CSVLogger(save_dir + '/log.csv')
##                es_cb = callbacks.EarlyStopping(monitor='val_output_loss', patience=20, verbose=1, mode='auto')
##    #                tb = callbacks.TensorBoard(log_dir=save_dir + '/tensorboard-logs',
##    #                                           batch_size=args.batch_size, histogram_freq=int(True))
##                checkpoint = callbacks.ModelCheckpoint(save_dir + '/weights-best.h5', monitor='val_output_accuracy',
##                                                       save_best_only=True, save_weights_only=True, verbose=1)
#    #                lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr_class * (0.9 ** epoch))
#                if not os.path.exists(save_dir):
#                        os.makedirs(save_dir)
#            
#        
#                rgb_train_dir = 'D:/CurtinFaces_crop/RGB/train/'
#                rgb_val_dir = 'D:/CurtinFaces_crop/RGB/test1/'
#    #            loaded_generator_model = keras.models.load_model('D:/DeepLearningImplementations/pix2pix/class_disc_add/models/CNN/gen_model.h5')
#                
#                generator_rgb = data_utils.flow_from_dir(rgb_train_dir,batch_size,img_dim)
#                
#                generator_rgb_val =  data_utils.flow_from_dir(rgb_val_dir,batch_size,img_dim)
#               
#            
#    #        
#    #            depth_guided_att.fit_generator(generator=generator_rgb,
#    #                steps_per_epoch=int(936 / int(batch_size)),##936 curtin faces###424 fold1 iiitd ##46846
#    #                epochs=nb_epoch_classes,
#    #                validation_data=generator_rgb_val,
#    #                validation_steps = int(2028 / int(batch_size)),##4108 curtin faces###4181 fold1 iiitd ##39942
#    #                callbacks=[log, checkpoint, es_cb])
#    #            
#                loss = 0
#                acc = 0
#                print('total_batches:',int(936 / batch_size))    
#                for x_batch_rgb, y_batch_rgb in generator_rgb:
#                    with graph.as_default():
#                        x_batch_depth = generator_model.predict(x_batch_rgb)
#                        if flip_counter == 0 and data_aug == 1:
#                            flip_img = iaa.Fliplr(1)(images=x_batch_rgb)
#                            rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_rgb)
#                            
#                            x_batch_rgb_final = np.concatenate([x_batch_rgb,flip_img,rot_img],axis=0)
#
#                            flip_img = iaa.Fliplr(1)(images=x_batch_depth)
#                            rot_img = iaa.Affine(rotate=(-30, 30))(images=x_batch_depth) 
#                            x_batch_depth_final = np.concatenate([x_batch_depth,flip_img,rot_img],axis=0)       
#                            
#                            y_batch_rgb_final = np.tile(y_batch_rgb,(3,1))
#                            
#                            flip_counter = 1
#
#                        elif flip_counter == 1 and data_aug == 1:
#
#                            shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_rgb)
#                            trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_rgb)
#                            x_batch_rgb_final = np.concatenate([x_batch_rgb,shear_aug,trans_aug],axis=0)
#
#
#                            shear_aug = iaa.Affine(shear=(-16, 16))(images=x_batch_depth)
#                            trans_aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})(images=x_batch_depth)
#                            x_batch_depth_final = np.concatenate([x_batch_depth,shear_aug,trans_aug],axis=0)
#                            
#                            y_batch_rgb_final = np.tile(y_batch_rgb,(3,1))
#                            flip_counter = 0 
#                        elif data_aug == 0:
#                            x_batch_rgb_final = x_batch_rgb
#                            x_batch_depth_final = x_batch_depth
#                            y_batch_rgb_final = y_batch_rgb
#                    
#                    out_loss = depth_guided_att.train_on_batch([x_batch_rgb_final,x_batch_depth_final], [y_batch_rgb_final, y_batch_rgb_final, y_batch_rgb_final]) 
#                    loss += out_loss[1]
#                    acc += out_loss[4]
#                    batch_counter += 1
#                    
#                    if batch_counter >= int(936 / int(batch_size)):
#                        break
##                    print('loss: {} ; Accuracy: {}'.format(out_loss[1],out_loss[4]))
##                    progbar.add(batch_size, values=['loss: {} ; Accuracy: {}'.format(out_loss[1],out_loss[4])])
#                print_loss = loss/int(936 / int(batch_size))
#                print_acc = acc/int(936 / int(batch_size))
#                print("")
#                print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch_classes, time.time() - start))  
#                print('loss: {} ; Accuracy: {}'.format(print_loss, print_acc))
#                val_loss = 0
#                val_acc1 = 0
#                val_acc2 = 0
#                val_acc3 = 0
#                batch_counter_val = 1
#                
##                steps = int(generator_rgb_val.samples/batch_size)
#                for x_batch_rgb, y_batch_rgb in generator_rgb_val:
#                    with graph.as_default():
#                        x_batch_depth = generator_model.predict(x_batch_rgb)
#
#                    out_loss = depth_guided_att.test_on_batch([x_batch_rgb, x_batch_depth], [y_batch_rgb, y_batch_rgb, y_batch_rgb])                      
#
#                    
#                    val_loss += out_loss[1]
#                    val_acc1 += out_loss[4]
#                    val_acc2 += out_loss[5]
#                    val_acc3 += out_loss[6]
#                    batch_counter_val += 1
##                    print('batch_counter:{}'.format(batch_counter))
#                    if batch_counter_val >= int(2028 / int(batch_size)):
#                        break
#                
#                print_val_loss = val_loss/int(2028 / int(batch_size))
#                print_val_acc1 = val_acc1/int(2028 / int(batch_size))
#                print_val_acc2 = val_acc2/int(2028 / int(batch_size))
#                print_val_acc3 = val_acc3/int(2028 / int(batch_size))
#                print('Val loss: {} ; Val Accuracy: {}, RGB:{}, Depth:{}'.format(print_val_loss, print_val_acc1, print_val_acc2, print_val_acc3))
#                patience_counter = 0
#                if val_loss < prev_val_loss:
#                    print('save weights')
#                    depth_guided_att_weights_path = os.path.join(save_dir, 'best_weights.h5')
##                    if not os.path.exists(depth_guided_att_weights_path):
##                        os.makedirs(depth_guided_att_weights_path)
#                    depth_guided_att.save_weights(depth_guided_att_weights_path, overwrite=True)  
#                    prev_val_loss = val_loss
#                else:
#                    patience_counter += 1 
#                    if patience_counter>5:
#                        break
#                
#            
    except KeyboardInterrupt:
        pass
