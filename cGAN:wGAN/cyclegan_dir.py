# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:56:26 2020

@author: hardi
"""

import os
import cv2
import random
from shutil import copyfile
from tqdm import tqdm 
import numpy as np
in_dir = 'D:/CurtinFaces_crop_gan/DEPTH/'
out_dir = 'D:/CurtinFaces_crop_gan/data/CurtinFaces/DEPTH'


#image_list = os.listdir(in_dir)
#
#test_set_images = random.sample(image_list, int(len(image_list)/10))

image_list = os.listdir(in_dir)
#subject_list = np.arange(1,53)
#test_set_subject = random.sample(list(subject_list), int(len(subject_list)/10))
#[15,39,48,17,22]
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    
for image in tqdm(image_list):
    image_path = os.path.join(in_dir,image)
    if int(image[0:2]) in test_set_subject:    
        image_path = os.path.join(in_dir,image)
        test_path = os.path.join(out_dir,'test')
        image_arr = cv2.imread(image_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        dest_path = os.path.join(test_path,image)
        image_arr = cv2.resize(image_arr,(256,256))
        cv2.imwrite(dest_path, image_arr)
#        copyfile(image_path,dest_path)
    
    
    else:
        train_path = os.path.join(out_dir,'train')
        image_arr = cv2.imread(image_path)
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        dest_path = os.path.join(train_path,image)
        image_arr = cv2.resize(image_arr,(256,256))
        cv2.imwrite(dest_path, image_arr)
#        copyfile(image_path,dest_path)
    
    
    