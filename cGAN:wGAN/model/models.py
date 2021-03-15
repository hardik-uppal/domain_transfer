from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
from keras.engine.topology import Layer
import numpy as np
from keras_vggface import utils
from keras.utils.data_utils import get_file
from attention_module import cbam_block, spatial_attention, channel_attention, mlb_attention, lstm_attention
import math

WIDTH = 128
HEIGHT = 128
TV_WEIGHT = math.pow(10, -6)

def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


def lambda_output(input_shape):
    return input_shape[:2]


# def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, dropout=False, strides=(2,2)):

#     x = Conv2D(f, (3, 3), strides=strides, name=name, padding="same")(x)
#     if bn:
#         x = BatchNormalization(axis=bn_axis)(x)
#     x = LeakyReLU(0.2)(x)
#     if dropout:
#         x = Dropout(0.5)(x)

#     return x


# def up_conv_block_unet(x1, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):

#     x1 = UpSampling2D(size=(2, 2))(x1)
#     x = merge([x1, x2], mode="concat", concat_axis=bn_axis)

#     x = Conv2D(f, (3, 3), name=name, padding="same")(x)
#     if bn:
#         x = BatchNormalization(axis=bn_axis)(x)
#     x = Activation("relu")(x)
#     if dropout:
#         x = Dropout(0.5)(x)

#     return x

def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, strides=(2,2)):

    x = LeakyReLU(0.2)(x)
    x = Conv2D(f, (3, 3), strides=strides, name=name, padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)

    return x


def up_conv_block_unet(x, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(f, (3, 3), name=name, padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])

    return x


def deconv_block_unet(x, x2, f, h, w, batch_size, name, bn_mode, bn_axis, bn=True, dropout=False):

    o_shape = (batch_size, h * 2, w * 2, f)
    x = Activation("relu")(x)
    x = Deconv2D(f, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])

    return x

def generator_unet_upsampling(img_dim, bn_mode, model_name="generator_unet_upsampling"):

    nb_filters = 64

    if K.image_data_format() == "channels_first":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                           strides=(2, 2), name="unet_conv2D_1", padding="same")(unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv_encoded = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv_encoded)

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(1, (3, 3), name="last_conv", padding="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(inputs=[unet_input], outputs=[x], name=model_name)

    return generator_unet

def generator_unet_upsampling_rev(img_dim, bn_mode, model_name="generator_unet_upsampling_rev"):

    nb_filters = 64

    if K.image_data_format() == "channels_first":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                           strides=(2, 2), name="unet_conv2D_1", padding="same")(unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv_encoded = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv_encoded)

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(3, (3, 3), name="last_conv", padding="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(inputs=[unet_input], outputs=[x], name=model_name)

    return generator_unet

def generator_unet_upsampling_2out(img_dim, bn_mode,n_class, model_name="generator_unet_upsampling_2_out"):

    nb_filters = 64

    if K.image_data_format() == "channels_first":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                           strides=(2, 2), name="unet_conv2D_1", padding="same")(unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv_encoded = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv_encoded)

    flat_model = Flatten(name='flatten_enc_GAN')(list_encoder[-1])
    
    fc6 = Dense(512, activation='relu', name='fc1_enc_GAN')(flat_model)
    bn_1 = BatchNormalization(name='1_bn')(fc6)
    fc7 = Dense(512, activation='relu', name='fc2_enc_GAN')(bn_1)
    bn_2 = BatchNormalization(name='2_bn')(fc7)
    
    output = Dense(n_class, activation='softmax', name='output_enc_GAN')(bn_2)  
    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(1, (3, 3), name="last_conv", padding="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(inputs=[unet_input], outputs=[x, output], name=model_name)

    return generator_unet


def generator_unet_deconv(img_dim, bn_mode, batch_size, model_name="generator_unet_deconv"):

    assert K.backend() == "tensorflow", "Not implemented with theano backend"

    nb_filters = 64
    bn_axis = -1
    h, w, nb_channels = img_dim
    min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                           strides=(2, 2), name="unet_conv2D_1", padding="same")(unet_input)]
    # update current "image" h and w
    h, w = h / 2, w / 2
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv)
        h, w = h / 2, w / 2

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-1][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [deconv_block_unet(list_encoder[-1], list_encoder[-2],
                                      list_nb_filters[0], h, w, batch_size,
                                      "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    h, w = h * 2, w * 2
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = deconv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, h,
                                 w, batch_size, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)
        h, w = h * 2, w * 2

    x = Activation("relu")(list_decoder[-1])
    o_shape = (batch_size,) + img_dim
    x = Deconv2D(nb_channels, (3, 3), output_shape=o_shape, strides=(2, 2), padding="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(inputs=[unet_input], outputs=[x])

    return generator_unet


def DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name="DCGAN_discriminator", use_mbd=True):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    list_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]

    if K.image_data_format() == "channels_first":
        bn_axis = 1
    else:
        bn_axis = -1

    nb_filters = 64
    nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
    list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv
    x_input = Input(shape=img_dim, name="discriminator_input")
    x = Conv2D(list_filters[0], (3, 3), strides=(2, 2), name="disc_conv2d_1", padding="same")(x_input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Next convs
    for i, f in enumerate(list_filters[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same")(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    x_flat = Flatten()(x)
    x = Dense(1, activation="linear", name="disc_dense")(x_flat)

    PatchGAN = Model(inputs=[x_input], outputs=[x, x_flat], name="PatchGAN")
    print("PatchGAN summary")
    PatchGAN.summary()

    x = [PatchGAN(patch)[0] for patch in list_input]
    x_mbd = [PatchGAN(patch)[1] for patch in list_input]

    if len(x) > 1:
        x = Concatenate(axis=bn_axis)(x)
    else:
        x = x[0]

    if use_mbd:
        if len(x_mbd) > 1:
            x_mbd = Concatenate(axis=bn_axis)(x_mbd)
        else:
            x_mbd = x_mbd[0]

        num_kernels = 100
        dim_per_kernel = 5

        M = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)
        MBD = Lambda(minb_disc, output_shape=lambda_output)

        x_mbd = M(x_mbd)
        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
        x_mbd = MBD(x_mbd)
        x = Concatenate(axis=bn_axis)([x, x_mbd])

    x_out = Dense(1, activation="linear", name="disc_output")(x)

    discriminator_model = Model(inputs=list_input, outputs=[x_out], name=model_name)

    return discriminator_model

def DCGAN_discriminator_cat(img_dim, nb_patch, bn_mode,n_class, model_name="DCGAN_discriminator_cat", use_mbd=True):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    list_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]

    if K.image_data_format() == "channels_first":
        bn_axis = 1
    else:
        bn_axis = -1

    nb_filters = 64
    nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
    list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv
    x_input = Input(shape=img_dim, name="discriminator_input")
    x = Conv2D(list_filters[0], (3, 3), strides=(2, 2), name="disc_conv2d_1", padding="same")(x_input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Next convs
    for i, f in enumerate(list_filters[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same")(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    x_flat = Flatten()(x)
    x = Dense(104, activation="softmax", name="disc_dense")(x_flat)

    PatchGAN = Model(inputs=[x_input], outputs=[x, x_flat], name="PatchGAN")
    print("PatchGAN summary")
    PatchGAN.summary()

    x = [PatchGAN(patch)[0] for patch in list_input]
    x_mbd = [PatchGAN(patch)[1] for patch in list_input]

    if len(x) > 1:
        x = Concatenate(axis=bn_axis)(x)
    else:
        x = x[0]

    if use_mbd:
        if len(x_mbd) > 1:
            x_mbd = Concatenate(axis=bn_axis)(x_mbd)
        else:
            x_mbd = x_mbd[0]

        num_kernels = 100
        dim_per_kernel = 5

        M = Dense(num_kernels * dim_per_kernel, use_bias=False, activation=None)
        MBD = Lambda(minb_disc, output_shape=lambda_output)

        x_mbd = M(x_mbd)
        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
        x_mbd = MBD(x_mbd)
        x = Concatenate(axis=bn_axis)([x, x_mbd])

    x_out = Dense(n_class, activation="softmax", name="disc_output")(x)

    discriminator_model = Model(inputs=list_input, outputs=[x_out], name=model_name)

    return discriminator_model

def DCGAN(generator, discriminator_model, img_dim, patch_size, image_dim_ordering):

    gen_input = Input(shape=img_dim, name="DCGAN_input")

    generated_image = generator(gen_input)

    if image_dim_ordering == "channels_first":
        h, w = img_dim[1:]
    else:
        h, w = img_dim[:-1]
    ph, pw = patch_size

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            if image_dim_ordering == "channels_last":
                x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
            else:
                x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriminator_model(list_gen_patch)

    DCGAN = Model(inputs=[gen_input],
                  outputs=[DCGAN_output],
                  name="DCGAN")

    return DCGAN

def CATGAN(generator, discriminator_model, img_dim, image_dim_ordering):

    gen_input = Input(shape=img_dim, name="CATGAN_input")

    generated_image = generator(gen_input)

    if image_dim_ordering == "channels_first":
        h, w = img_dim[1:]
    else:
        h, w = img_dim[:-1]
#    ph, pw = patch_size
#
#    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
#    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]
#
#    list_gen_patch = []
#    for row_idx in list_row_idx:
#        for col_idx in list_col_idx:
#            if image_dim_ordering == "channels_last":
#                x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
#            else:
#                x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
#            list_gen_patch.append(x_patch)

    CATGAN_output = discriminator_model([gen_input,generated_image])

    CATGAN = Model(inputs=[gen_input],
                  outputs=[generated_image, CATGAN_output],
                  name="CATGAN")

    return CATGAN

def encoder_GAN(generator, discriminator_model, n_class, img_dim, patch_size, image_dim_ordering):

    gen_input = Input(shape=img_dim, name="en_gen_input")

#    encoded_image, class_out = encoder(gen_input)
    generated_image, conv_encoded = generator(gen_input)
    flat_model = Flatten(name='flatten_enc_GAN')(conv_encoded)
    
    fc6 = Dense(512, activation='relu', name='fc1_enc_GAN')(flat_model)
    bn_1 = BatchNormalization(name='1_bn')(fc6)
    fc7 = Dense(512, activation='relu', name='fc2_enc_GAN')(bn_1)
    bn_2 = BatchNormalization(name='2_bn')(fc7)
    
    output = Dense(n_class, activation='softmax', name='output_enc_GAN')(bn_2)    
    
    if image_dim_ordering == "channels_first":
        h, w = img_dim[1:]
    else:
        h, w = img_dim[:-1]
    ph, pw = patch_size

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h // ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w // pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            if image_dim_ordering == "channels_last":
                x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
            else:
                x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
            list_gen_patch.append(x_patch)

    GAN_output = discriminator_model(list_gen_patch)

    GAN = Model(inputs=[gen_input],
                  outputs=[generated_image, GAN_output, output],
                  name="CATGAN")

    return GAN


def vgg_encoder(input_shape,n_class, include_top=True):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
   
    
    :return:  Keras Model used for training
    """
    # RGB MODALITY BRANCH OF CNN
#    if input_tensor is None:
    inputs_rgb = Input(shape=input_shape, name='input')
#    else:
#        img_input = input_tensor    

    ########################VGG/RESNET or any other network
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
        inputs_rgb)
#    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    pool1_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
        pool1_rgb)
#    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
#        x)
    pool2_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
        pool2_rgb)
#    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
#        x)
#    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
#        x)
    pool3_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
        pool3_rgb)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
#        x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
#        x)
    pool4_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
        pool4_rgb)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
#        x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
#        x)
    conv_model_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
#    vgg_model_rgb = VGG16(include_top=False, weights='vggface', input_tensor=None, input_shape=input_shape, pooling=None, type_name='rgb')
#    conv_model_rgb = vgg_model_rgb(inputs_rgb)
#    attention_spatial_rgb = spatial_attention(conv_model_rgb)
    #load weights for vggFace
#    rgb_model = Model(inputs=[inputs_rgb], outputs=[conv_model_rgb])
#    weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5', utils.VGG16_WEIGHTS_PATH_NO_TOP, cache_subdir=utils.VGGFACE_DIR)
#    rgb_model.load_weights(weights_path)
    flat_model = Flatten(name='flatten')(conv_model_rgb)
    if include_top:
                
        fc6 = Dense(2048, activation='relu', name='fc6')(flat_model)
        bn_1 = BatchNormalization(name='1_bn')(fc6)
        fc7 = Dense(1024, activation='relu', name='fc7')(bn_1)
        bn_2 = BatchNormalization(name='2_bn')(fc7)
        
        output = Dense(n_class, activation='softmax', name='output')(bn_2)    
        
        model = Model(inputs=[inputs_rgb], outputs=[output])
    else:
        model = Model(inputs=[inputs_rgb], outputs=[flat_model])
    
    return model

def depth_guided_att(input_shape, n_class):
    """
    
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    
    :return:  Keras Model used for training
    """
    # RGB MODALITY BRANCH OF CNN
    inputs_rgb = Input(shape=input_shape, name='input_rgb')
    ########################VGG/RESNET or any other network
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1_rgb')(
        inputs_rgb)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2_rgb')(x)
    pool1_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_rgb')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1_rgb')(
        pool1_rgb)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2_rgb')(
        x)
    pool2_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_rgb')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1_rgb')(
        pool2_rgb)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2_rgb')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3_rgb')(
        x)
    pool3_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool3_rgb')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1_rgb')(
        pool3_rgb)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2_rgb')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3_rgb')(
        x)
    pool4_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool4_rgb')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1_rgb')(
        pool4_rgb)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2_rgb')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3_rgb')(
        x)
    conv_model_rgb = MaxPooling2D((2, 2), strides=(2, 2), name='pool5_rgb')(x)
#    vgg_model_rgb = VGG16(include_top=False, weights='vggface', input_tensor=None, input_shape=input_shape, pooling=None, type_name='rgb')
#    conv_model_rgb = vgg_model_rgb(inputs_rgb)
#    attention_spatial_rgb = spatial_attention(conv_model_rgb)
    #load weights for vggFace
    rgb_model = Model(inputs=[inputs_rgb], outputs=[conv_model_rgb])
    weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5', utils.VGG16_WEIGHTS_PATH_NO_TOP, cache_subdir=utils.VGGFACE_DIR)
    rgb_model.load_weights(weights_path)


    inputs_depth = Input(shape=input_shape, name = "inputs_depth")
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1_depth')(
        inputs_depth)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2_depth')(x)
    pool1_depth = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_depth')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1_depth')(
        pool1_depth)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2_depth')(
        x)
    pool2_depth = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_depth')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1_depth')(
        pool2_depth)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2_depth')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3_depth')(
        x)
    pool3_depth = MaxPooling2D((2, 2), strides=(2, 2), name='pool3_depth')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1_depth')(
        pool3_depth)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2_depth')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3_depth')(
        x)
    pool4_depth = MaxPooling2D((2, 2), strides=(2, 2), name='pool4_depth')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1_depth')(
        pool4_depth)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2_depth')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3_depth')(
        x)
    conv_model_depth = MaxPooling2D((2, 2), strides=(2, 2), name='pool5_depth')(x)
#    vgg_model_depth = VGG16(include_top=False, weights='vggface', input_tensor=None, input_shape=input_shape, pooling=None, type_name='depth')
#    conv_model_depth = vgg_model_depth(inputs_depth)
    #load weights for vggFace
    depth_model = Model(inputs=[inputs_depth], outputs=[conv_model_depth])
    weights_path = get_file('rcmalli_vggface_tf_notop_vgg16.h5', utils.VGG16_WEIGHTS_PATH_NO_TOP, cache_subdir=utils.VGGFACE_DIR)
    depth_model.load_weights(weights_path)
    blk1_depth = MaxPooling2D((33, 33), strides=(16, 16),padding='same', name='blk1_depth')(pool1_depth)
    blk2_depth = MaxPooling2D((17, 17), strides=(8, 8),padding='same', name='blk2_depth')(pool2_depth)
    blk3_depth = MaxPooling2D((9, 9), strides=(4, 4),padding='same', name='blk3_depth')(pool3_depth)
    blk4_depth = MaxPooling2D((3, 3), strides=(2, 2),padding='same', name='blk4_depth')(pool4_depth)
    
    
    
    mfcc_depth = concatenate([blk1_depth,blk2_depth,blk3_depth,blk4_depth,conv_model_depth], axis=-1)
#    merge_rgb_depth = layers.concatenate([mfcc_rgb,mfcc_depth], axis=-1)
#    sav_depth = channel_attention(mfcc_depth)
    
    
#    attention_spatial_depth = spatial_attention_weights(conv_model_depth)


    # CONACTENATE the ends of RGB & DEPTH 

#    merge_rgb_depth = layers.concatenate([conv_model_rgb,conv_model_depth], axis=-1)
#    merge_rgb_depth = layers.concatenate([attention_spatial_rgb,attention_spatial_depth], axis=-1)
 
    
    
    ### Attention mechanism
    
    
#    primarycaps = PrimaryCap(merge_rgb_depth, dim_capsule=16, n_channels=32, kernel_size=3, strides=1, padding='valid')
#    secondarycaps = PrimaryCap(primarycaps, dim_capsule=8, n_channels=32, kernel_size=3, strides=1, padding='valid')
#    idcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=32, routings=3, name='idcaps')(primarycaps)
#    attention_features = channel_attention(conv_model_rgb)
#    attention_features = cbam_block(conv_model_rgb)
    ## new attention mech
#    new_att_features= multiply([conv_model_rgb, conv_model_depth])
#  
#    attention_features = cbam_block(merge_rgb_depth)
    
    
    attention_features = mlb_attention(conv_model_rgb,mfcc_depth, ratio=[8,2])
#
    

    ######## Common network
    flat_model = Flatten(name='flatten')(attention_features)
    flat_model_rgb = Flatten(name='flatten_rgb')(conv_model_rgb)
    flat_model_depth = Flatten(name='flatten_depth')(conv_model_depth)

    fc6 = Dense(1024, activation='relu', name='fc6')(flat_model_rgb)
    bn_1 = BatchNormalization(name='1_bn')(fc6)
#    dropout_1 = layers.Dropout(0.5)(bn_1)
#    
#    
#    
#
#
    fc7 = Dense(1024, activation='relu', name='fc7')(flat_model_depth)
    bn_2 = BatchNormalization(name='2_bn')(fc7)
#    dropout_2 = layers.Dropout(0.5)(bn_2)
    
    fc8 = Dense(2048, activation='relu', name='fc8')(flat_model)
    bn_3 = BatchNormalization(name='3_bn')(fc8)
#    dropout_3 = layers.Dropout(0.5)(bn_3)
    

    
    #VECTORIZING OUTPUT
    output = Dense(n_class, activation='softmax', name='output')(bn_3)
    output_rgb = Dense(n_class, activation='softmax', name='output_rgb')(bn_1)
    output_depth = Dense(n_class, activation='softmax', name='output_depth')(bn_2)
    
    # MODAL [INPUTS , OUTPUTS]
    train_model = Model(inputs=[inputs_rgb, inputs_depth], outputs=[output,output_rgb,output_depth])
    
#    weights_path = 'CurtinFaces/vgg_multimodal_dropout-0.5_3fc-512/weights-25.h5'
#    train_model.load_weights(weights_path)
#    train_model.summary()
#    for layer in train_model.layers[:37]:
#        layer.trainable = False
#    for layer in train_model.layers[11]:
#    train_model.layers[11].trainable = False
##    for layer in train_model.layers[14]:
#    train_model.layers[11].trainable = False
#    for layer in train_model.layers[2].layers[:-4]:
#        layer.trainable = False
#    for layer in train_model.layers[3].layers[:-4]:
#        layer.trainable = False




    return train_model

def get_content_loss(args):
    new_activation, content_activation = args[0], args[1]
    return K.mean(K.square(new_activation - content_activation))


def gram_matrix(activation):
    
    assert K.ndim(activation) == 3
    shape = K.shape(activation)
#    print(activation)
#    print(shape[0],shape[1], shape[2])
    shape = (shape[0] * shape[1], shape[2])
    
    # reshape to (H*W, C)
    activation = K.reshape(activation, shape)
    shape = K.cast(shape, dtype='float32')
    return K.dot(K.transpose(activation), activation) / (shape[0] * shape[1])


def get_style_loss(args):
    new_activation, style_activation = args[0], args[1]
    original_gram_matrix = gram_matrix(style_activation[0])

    new_gram_matrix = gram_matrix(new_activation[0])
    return K.sum(K.square(original_gram_matrix - new_gram_matrix))


def get_TV(new_gram_matrix):
    x_diff = K.square(new_gram_matrix[:, :WIDTH - 1, :HEIGHT - 1, :] - new_gram_matrix[:, 1:, :HEIGHT - 1, :])
    y_diff = K.square(new_gram_matrix[:, :WIDTH - 1, :HEIGHT - 1, :] - new_gram_matrix[:, :WIDTH - 1, 1:, :])
    return TV_WEIGHT * K.mean(K.sum(K.pow(x_diff + y_diff, 1.25)))


def get_vgg_activation(model, tensor, layer_name):
    input_tensor = Input(tensor=tensor, shape=tensor.shape)
    pred_depth = model(input_tensor)
    model_edit = Model(inputs=[input_tensor], outputs=[pred_depth])
#    model = vgg16.VGG16(input_tensor=input_tensor, input_shape=(256, 256, 3), weights='imagenet', include_top=False)
    outputs_dict = {}
    for layer in model_edit.layers:
        outputs_dict[layer.name] = layer.output
        layer.trainable = False
    return outputs_dict[layer_name]


def dummy_loss_function(y_true, y_pred):
    return y_pred


def zero_loss_function(y_true, y_pred):
    return K.variable(np.zeros(1,))

def build_graph_perceptual_loss(generator, classifier_model, img_dim):
    
    
    rgb_input = Input(shape=img_dim, name="rgb_input")
#    depth_input = Input(shape=img_dim_depth, name="depth_input")
    
    generated_depth = generator(rgb_input) 

    content_activation = Input(shape=(64, 64, 128))
    style_activation1 = Input(shape=(128, 128, 64))
    style_activation2 = Input(shape=(64, 64, 128))
    style_activation3 = Input(shape=(32, 32, 256))
    style_activation4 = Input(shape=(16, 16, 512))

    total_variation_loss = Lambda(get_TV, output_shape=(1,), name='tv')(generated_depth)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(generated_depth)
#    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    style_loss1 = Lambda(get_style_loss, output_shape=(1,), name='style1')([x, style_activation1])

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
        x)
#    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
#        x)
#    content_Loss = Lambda(get_content_loss, output_shape=(1,), name='content')([x, content_activation])
    style_loss2 = Lambda(get_style_loss, output_shape=(1,), name='style2')([x, style_activation2])

    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
        x)
#    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
#        x)
#    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
#        x)
    style_loss3 = Lambda(get_style_loss, output_shape=(1,), name='style3')([x, style_activation3])
    
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
        x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
#        x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
#        x)
    style_loss4 = Lambda(get_style_loss, output_shape=(1,), name='style4')([x, style_activation4])
    
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
        x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
#        x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
#        x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    model = Model(
#        [rgb_input, content_activation, style_activation1, style_activation2, style_activation3, style_activation4],
        [rgb_input, style_activation1, style_activation2, style_activation3, style_activation4],
#        [content_Loss, style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, generated_depth],name='generator_train')
        [style_loss1, style_loss2, style_loss3, style_loss4, total_variation_loss, generated_depth],name='generator_train')
    model_layers = {layer.name: layer for layer in model.layers}
    #load pre-trained classfier model for weights
#    original_vgg = classifier_model
    original_vgg_layers = {layer.name: layer for layer in classifier_model.layers}

    # load weight
    for layer in classifier_model.layers:
        if layer.name in model_layers:
            print(layer.name, model_layers[layer.name].output_shape)
            model_layers[layer.name].set_weights(original_vgg_layers[layer.name].get_weights())
            model_layers[layer.name].trainable = False

    print("VGG model built successfully!")
    return model    

def load(model_name, img_dim, nb_patch, bn_mode, use_mbd, batch_size, do_plot, n_class=None):

    if model_name == "generator_unet_upsampling":
        model = generator_unet_upsampling(img_dim, bn_mode, model_name=model_name)
        model.summary()
        if do_plot:
            from keras.utils import plot_model
            plot_model(model, to_file="../../figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model


#generator_unet_upsampling_2out
#    n_class=52
    if model_name == "generator_unet_upsampling_2out":
        model = generator_unet_upsampling_2out(img_dim, bn_mode,n_class, model_name=model_name)
        model.summary()
        if do_plot:
            from keras.utils import plot_model
            plot_model(model, to_file="../../figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model
    
    if model_name == "generator_unet_deconv":
        model = generator_unet_deconv(img_dim, bn_mode, batch_size, model_name=model_name)
        model.summary()
        if do_plot:
            from keras.utils import plot_model
            plot_model(model, to_file="../../figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "DCGAN_discriminator":
        model = DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name=model_name, use_mbd=use_mbd)
        model.summary()
        if do_plot:
            from keras.utils import plot_model
            plot_model(model, to_file="../../figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "DCGAN_discriminator_cat":
        model = DCGAN_discriminator_cat(img_dim, nb_patch, bn_mode, model_name=model_name, use_mbd=use_mbd)
        model.summary()
        if do_plot:
            from keras.utils import plot_model
            plot_model(model, to_file="../../figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "depth_guided_att":
        model = depth_guided_att(img_dim, n_class=52)
        model.summary()
        if do_plot:
            from keras.utils import plot_model
            plot_model(model, to_file="../../figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model
    
    if model_name == "vgg_encoder":
        model = vgg_encoder(img_dim,n_class=52)
        model.summary()
        if do_plot:
            from keras.utils import plot_model
            plot_model(model, to_file="../../figures/%s.png" % model_name, show_shapes=True, show_layer_names=True)
        return model
    
if __name__ == "__main__":

    # load("generator_unet_deconv", (256, 256, 3), 16, 2, False, 32)
    load("generator_unet_upsampling", (256, 256, 3), 16, 2, False, 32)
