# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 04:46:05 2020

@author: hardi
"""

import cv2
import sys
import os
import models as model_lib
from keras.optimizers import Adam
from depth_metrics import depth_metrics
import numpy as np
sys.path.append("../utils")
#import general_utils
import data_utils
#logging_dir = '/test_metrics'
weight_path = 'D:/tutorial/hulk_dump/results/unet_mae_cat_loss/gen_weights_epoch200.h5'
weight_path = 'D:/tutorial/hulk_dump/results/unet_2out_gan/gen_weights_epoch30.h5'
#weight_path = 'D:/tutorial/hulk_dump/results/unet_mse_gan/gen_weights_epoch50.h5'
#if not os.path.exists(logging_dir):
#    os.mkdir(logging_dir)

generator = '_upsampling_2out'
img_dim =(128,128,3)
nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, (64,64), 'channels_last')
bn_mode = 2
use_mbd = True
batch_size = 4
do_plot = False



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



rgb_train_dir1 = 'D:/CurtinFaces_crop/RGB/train'
rgb_train_dir2 = 'D:/CurtinFaces_crop/RGB/test'
dataset = 'curtinfaces'

image_dir = [rgb_train_dir1,rgb_train_dir2]
#    generator_both_train = data_utils.flow_from_both_dir(rgb_train_dir, batch_size, img_dim,'pandora','train')
#data_gen_train = data_utils.data_pipeline(image_dir, dataset, test_train='train')
data_gen_test = data_utils.data_pipeline(image_dir, dataset, test_train='test')


generator_model = model_lib.load("generator_unet%s" % generator,
                              img_dim,
                              nb_patch,
                              bn_mode,
                              use_mbd,
                              batch_size,
                              do_plot,n_class=48)
opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

generator_model.compile(loss='mae', optimizer=opt_discriminator)


save_img = 1
save_path = 'D:/DeepLearningImplementations/pix2pix/depth_est_samples/unet_cat_loss_gan/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

generator_model.load_weights(weight_path)
metric_dict = {
        "L1":0,
        "L2":0,
        "ABS relative difference":0,
        "Sqrt relative difference":0,
        "RMSE(linear)":0,
        "RMSE(log)":0,
        "RMSE(log scale inv)":0,
        "Threshold(1.25)":0,
        "Threshold(1.25**2)":0,
        "Threshold(1.25**3)":0
    }

#metric_dict= {}
#metric_dict.fromkeys(metrics, 0)
test_batches = int(len(data_gen_test.image_test_arr)/batch_size)
for i in range(test_batches):
    x_rgb, x_depth, y_label = next(data_gen_test.flow_from_dir(batch_size, img_dim))
    gen_depth,pred_class = generator_model.predict(x_rgb)
#    gen_depth = np.array(gen_depth)
    
    gen_depth = inverse_normalization(gen_depth)
    x_depth = inverse_normalization(x_depth)
    x_rgb = inverse_normalization(x_rgb)

    metric_name, val = depth_metrics(x_depth, gen_depth)
    
    for metric,value in zip(metric_name, val):
        metric_dict[metric] = metric_dict[metric] + value
    
    if save_img:    
        image_path_rgb = save_path +'/sample_{}.jpg'.format(i)
        merge_img_rgb = cv2.hconcat([x_rgb[0], x_rgb[1], x_rgb[2], x_rgb[3]])
        cv2.imwrite(image_path_rgb,merge_img_rgb)
        
        image_path_depth = save_path +'/sample_depth_{}.jpg'.format(i)
        merge_img_depth = cv2.hconcat([x_depth[0], x_depth[1], x_depth[2], x_depth[3]])
    #    cv2.imwrite(image_path_depth,merge_img_depth)
        
    #    image_path_gen_depth = save_path +'/sample_gen_depth_{}.jpg'.format(i)
        merge_img_gen_depth = cv2.hconcat([gen_depth[0], gen_depth[1], gen_depth[2], gen_depth[3]])
        merge_img = cv2.vconcat([merge_img_depth, merge_img_gen_depth])
        
        cv2.imwrite(image_path_depth,merge_img)    

for key,val in metric_dict.items():
    metric_dict[key] = val/test_batches

   