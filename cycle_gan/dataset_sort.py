#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:34:01 2020

@author: hardikuppal
"""

import os
import pylib as py
from shutil import copyfile

rgb_dir = '/home/harry/Face_rec/lfw/RGB'
dataset_dir = '/home/harry/Face_rec/lfw/est_depth'

out_dir = '/home/harry/Face_rec/lfw/depth_new'
py.mkdir(out_dir)

for label in os.listdir(rgb_dir):
    subject_dir = py.join(out_dir, label)
    py.mkdir(subject_dir)
    
    for image in os.listdir(dataset_dir):
        if label in image:
            source_image_path = py.join(dataset_dir,image)
            
            dest_image_path = py.join(subject_dir,image)
            copyfile(source_image_path, dest_image_path)
    