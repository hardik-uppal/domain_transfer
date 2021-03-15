#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:09:40 2021

@author: hardikuppal
"""

import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module

#TODO
#ADD reverse hubber loss
# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='CurtinFaces')
py.arg('--dataset_target', default='imdb_face')
py.arg('--datasets_dir', default='/home/harry/Face_rec/CurtinFaces_crop/')
py.arg('--datasets_dir_target', default='/home/harry/Face_rec/imbdface/')
py.arg('--load_size', type=int, default=128)  # load image to this size
py.arg('--crop_size', type=int, default=128)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay_teach', type=int, default=25)  # epoch to start decaying learning rate
py.arg('--epoch_decay', type=int, default=50)
py.arg('--lr', type=float, default=0.000002)
py.arg('--lr_teach', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--MAE_teacher_loss_weight', type=float, default=10)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

#
##args.dataset = 'rgb'
## output_dir
#output_dir = py.join(args.dataset_target+'_depth_est_imdb', args.dataset)
#py.mkdir(output_dir)
#
## save settings
#py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================
## Generator for Student
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_A2B.summary()
## Generator for Teacher
#G_A2B_teacher = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
#G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_A.summary()