#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from keras.callbacks import Callback


class DecoderSnapshot(Callback):

    def __init__(self, step_size=500):
        super().__init__()
        self._step_size = step_size
        self._steps = 0
        self._epoch = 0
#        self._latent_dim = latent_dim
#        self._decoder_index = decoder_index
        self._img_rows = 128
        self._img_cols = 128
        self._thread_pool = ThreadPoolExecutor(1)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._steps = 0

    def on_batch_end(self, batch, logs=None):
#        print('\n')
#        print(len(batch))
        self._steps += 1
        if self._steps % self._step_size == 0:
            
#            try:
            if batch:
                rgb_input = batch[0]
                depth_input = batch[1]
#                print('\n input_shape:{},{}'.format(rgb_input.shape,depth_input.shape))
                self.plot_images(rgb_input = rgb_input, depth_input = depth_input, val = 'train')
#            except:
#                print('skipped batch print')
#                pass


    def on_predict_begin(self, batch, logs=None):
#        if self._steps % self._step_size == 0:
        try:
            print('saving generated val images')
#            if batch[0]:
            rgb_input = batch[0]
            depth_input = batch[1]
            self.plot_images(rgb_input = rgb_input, depth_input = depth_input, val='validation')
        except:
            print('generated val images failed..')
            pass


    def plot_images(self, rgb_input=None, depth_input=None, val=None):
        generator = self.model
        
        
        
#        z = np.random.normal(size=(samples, self._latent_dim))
#        print('\n rgb_input_shape:{}'.format(rgb_input.shape))
        if rgb_input is not None:
            
            filename = 'test_images1/%s_generated_%d_%d.png' % (val, self._epoch,self._steps)
            images = generator.predict(rgb_input)
#            print('\n input_shape:{}'.format(images.shape))
            self._thread_pool.submit(self.save_plot, images, filename)
        
            filename_rgb = 'test_images1/%s_rgb_%d_%d.png' % (val, self._epoch,self._steps)
            self._thread_pool.submit(self.save_plot, rgb_input, filename_rgb)
#        if depth_input:
            filename_depth = 'test_images1/%s_depth_%d_%d.png' % (val, self._epoch,self._steps)        
            self._thread_pool.submit(self.save_plot, depth_input, filename_depth)

    @staticmethod
    def save_plot(images, filename):
        images = (images + 1.) * 127.5
        images = np.clip(images, 0., 255.)
        np.rollaxis(images, 3, 1)  
        images = images.astype('uint8')
        rows = []
        for i in range(0, len(images), 4):
            rows.append(np.concatenate(images[i:(i + 4), :, :, :], axis=0))
        plot = np.concatenate(rows, axis=1).squeeze()
        Image.fromarray(plot).save(filename)


class ModelsCheckpoint(Callback):

    def __init__(self, epoch_format, *models):
        super().__init__()
        self._epoch_format = epoch_format
        self._models = models

    def on_epoch_end(self, epoch, logs=None):
        
        print('saving models..')
        suffix = self._epoch_format.format(epoch=epoch + 1, **logs)
        for model in self._models:
            model.save_weights('model_save1/'+model.name + suffix)
