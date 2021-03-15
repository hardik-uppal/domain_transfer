# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 04:06:32 2020

@author: hardi
"""

from __future__ import division, print_function
import numpy as np
from glob import glob
import cv2
import os
import sklearn
#from sklearn.metrics import mean_squared_error


#if __name__ == '__main__':
#    dataset_path = '/gt-dataset-path'
#    generated_dataset_paths = [
#        '/your-path-to-generated-images',
#    ]
def depth_metrics(original_image, generated_image):
    epsilon = 1e-10

    val_metrics = []

#    X_rgb_batch, X_depth_batch, y_label = next(data_gen)
#    
#    original_image = X_depth_batch
#    generated_image = generator.predict(X_rgb_batch)

    original_image = original_image.astype(np.float64)
    generated_image = generated_image.astype(np.float64)

    l1 = np.mean(np.abs(generated_image - original_image))
    l2 = np.linalg.norm(generated_image - original_image)

    original_image[original_image == 0] = 1
    abs_rel_diff = np.mean(np.abs(generated_image - original_image) / (original_image + epsilon))

    sqrt_rel_diff = np.mean(np.square(generated_image - original_image) / (original_image + epsilon))
    original_image[original_image == 1] = 0

    # the square root is done later
    rmse_lin = np.square(generated_image - original_image).mean()
    rmse_log = np.square(np.log(generated_image + epsilon) - np.log(original_image + epsilon)).mean()

    d = np.log(generated_image + epsilon) - np.log(original_image + epsilon)
    n = original_image.shape[0] * original_image.shape[1]
    rmse_log_si = np.sum(np.square(d)) / n - np.square(np.sum(d)) / (n ** 2)

    img_max = np.maximum((original_image / (generated_image + epsilon)), (generated_image / (original_image + epsilon)))
    th = 1.25

    p_th = np.sum((img_max < th).astype(np.uint8)) / n
    p_th2 = np.sum((img_max < (th ** 2)).astype(np.uint8)) / n
    p_th3 = np.sum((img_max < (th ** 3)).astype(np.uint8)) / n

    frame_metric = np.array(
        [l1, l2, abs_rel_diff, sqrt_rel_diff, rmse_lin, rmse_log, rmse_log_si, p_th, p_th2, p_th3])
    val_metrics.append(frame_metric)
    
    val_metrics = np.asarray(val_metrics)
    
    mean_metrics = np.mean(val_metrics, axis=0)
    mean_metrics[4:7] = np.sqrt(mean_metrics[4:7])  # square root of RMSEs
    
#    print("\n###########################################################################################################")
#    print(fd, current_dataset_dir)
#    print("###########################################################################################################")
    metrics = [
        "L1",
        "L2",
        "ABS relative difference",
        "Sqrt relative difference",
        "RMSE(linear)",
        "RMSE(log)",
        "RMSE(log scale inv)",
        "Threshold(1.25)",
        "Threshold(1.25**2)",
        "Threshold(1.25**3)"
    ]
#    for m, val in zip(metrics, mean_metrics):
#        print("{:>25}: ".format(m), val)
    
#    print("###########################################################################################################\n")
    return metrics, mean_metrics