import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Concatenate
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Reshape, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Deconv2D, UpSampling2D
from keras.layers.pooling import GlobalAveragePooling2D


def wasserstein(y_true, y_pred):

    # return K.mean(y_true * y_pred) / K.mean(y_true)
    return K.mean(y_true * y_pred)


def conv_block_unet(x, f, name, bn_axis, bn=True, strides=(2,2)):

    x = LeakyReLU(0.2)(x)
    x = Conv2D(f, (3, 3), strides=strides, name=name, padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)

    return x

def up_conv_block_unet(x,  f, name, bn_axis, bn=True, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(f, (3, 3), name=name, padding="same")(x)
    if bn:
        x = BatchNormalization(axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
#    x = Concatenate(axis=bn_axis)([x, x2])

    return x

def generator_upsampling(noise_dim, img_dim, n_class, model_name="generator_upsampling"):
    """DCGAN generator based on Upsampling and Conv2D

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        model_name: model name (default: {"generator_upsampling"})
        

    Returns:
        keras model
    """

    if K.image_data_format() == "channels_first":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])        

    # Prepare encoder filters
    nb_filters = 64    
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
    
#    noise_input = Input(shape=noise_dim, name="noise_input")
    image_input = Input(shape=img_dim, name="image_input")
    list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                           strides=(2, 2), name="unet_conv2D_1", padding="same")(image_input)]    
    for i,f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
#        x = UpSampling2D(size=(2, 2))(x)
#        nb_filters = int(f / (2 ** (i + 1)))
        conv_encoded = conv_block_unet(list_encoder[-1], f, name, bn_axis)
        list_encoder.append(conv_encoded)   


    flat_model = Flatten(name='flatten_enc_GAN')(list_encoder[-1])
    
    fc6 = Dense(512, activation='relu', name='fc1_enc_GAN')(flat_model)
    bn_1 = BatchNormalization(name='1_bn')(fc6)
    fc7 = Dense(512, activation='relu', name='fc2_enc_GAN')(bn_1)
    bn_2 = BatchNormalization(name='2_bn')(fc7)
    
    output = Dense(n_class, activation='softmax', name='output_enc_GAN')(bn_2)      
    # Noise input and reshaping
   
#    conv_noise = Concatenate(axis=bn_axis)([list_encoder[-1],noise_input])

    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [up_conv_block_unet(list_encoder[-1],list_nb_filters[0], "unet_upconv2D_1",  bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], f, name, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(1, (3, 3), name="last_conv", padding="same")(x)
    x = Activation("tanh",name='generated_image_out')(x)

    generator_model = Model(inputs=[image_input], outputs=[x, output], name = model_name)
    generator_model.summary()
    return generator_model

##two functions added for patches GAN discriminator
def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


def lambda_output(input_shape):
    return input_shape[:2]

def discriminator(img_dim, bn_mode, nb_patch, use_mbd, model_name="discriminator"):
    """DCGAN discriminator

    Args:
        img_dim: dimension of the image output
        bn_mode: keras batchnorm mode
        model_name: model name (default: {"generator_deconv"})

    Returns:
        keras model
    """

    if K.image_data_format() == "channels_first":
        bn_axis = 1
        nb_channels = 1
        img_dim = list(img_dim)
        img_dim[0] = 1
        img_dim = tuple(img_dim)
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = 1
        img_dim = list(img_dim)
        img_dim[-1] = 1
        img_dim = tuple(img_dim)
        min_s = min(img_dim[:-1])  

#    disc_input = Input(shape=img_dim, name="discriminator_input")
    
    disc_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]        

    # Get the list of number of conv filters
    # (first layer starts with 64), filters are subsequently doubled
    nb_conv = int(np.floor(np.log(min_s // 4) / np.log(2)))
    list_f = [64 * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv with 2x2 strides
    x_input = Input(shape=img_dim, name="discriminator_input")
    x = Conv2D(list_f[0], (3, 3), strides=(2, 2), name="disc_conv2d_1",
               padding="same", use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.02))(x_input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Conv blocks: Conv2D(2x2 strides)->BN->LReLU
    for i, f in enumerate(list_f[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same", use_bias=False,
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    # Last convolution
    x = Conv2D(1, (3, 3), name="last_conv", padding="same", use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.02))(x)
    
    
    #added for patches
    x_flat = Flatten()(x)
    # Average pooling
    x = GlobalAveragePooling2D(name="out_disc")(x)

    PatchGAN = Model(inputs=[x_input], outputs=[x, x_flat], name="PatchGAN_wgan")
    print("PatchGAN summary")
    PatchGAN.summary()

    x = [PatchGAN(patch)[0] for patch in disc_input]
    x_mbd = [PatchGAN(patch)[1] for patch in disc_input]

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
    # Average pooling
    x_out = Dense(1, activation="relu", name="disc_output")(x)

    discriminator_model = Model(inputs=disc_input, outputs=[x_out], name=model_name)
#    visualize_model(discriminator_model)
    discriminator_model.summary()
    return discriminator_model


def DCGAN(generator, discriminator, noise_dim, img_dim):
    """DCGAN generator + discriminator model

    Args:
        generator: keras generator model
        discriminator: keras discriminator model
        noise_dim: generator input noise dimension
        img_dim: real image data dimension

    Returns:
        keras model
    """

#    noise_input = Input(shape=noise_dim, name="noise_input_dcgan")
    image_input = Input(shape=img_dim, name="image_input_dcgan")
    generated_image = generator([image_input])
    DCGAN_output = discriminator(generated_image)

    DCGAN = Model(inputs=[image_input],
                  outputs=[DCGAN_output,generated_image],
                  name="DCGAN")
    DCGAN.summary()

    return DCGAN



def DCGAN_wgan(generator, discriminator_model, noise_dim, img_dim, patch_size, image_dim_ordering):

#    noise_input = Input(shape=noise_dim, name="noise_input_dcgan")
    image_input = Input(shape=img_dim, name="image_input_dcgan")
    generated_image, pred_class = generator([image_input])

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

    DCGAN = Model(inputs=[image_input],
                  outputs=[DCGAN_output, generated_image, pred_class],
                  name="DCGAN_wgan")
    DCGAN.summary()

    return DCGAN

def encoder_vae(noise_dim, img_dim, n_class, model_name="encoder"):
    """encoder 

    Args:
        noise_dim: Dimension of the noise input
        img_dim: dimension of the image output
        model_name: model name (default: {"generator_upsampling"})
        

    Returns:
        keras model
    """

    if K.image_data_format() == "channels_first":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])        

    # Prepare encoder filters
    nb_filters = 64    
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
    
#    noise_input = Input(shape=noise_dim, name="noise_input")
    image_input = Input(shape=img_dim, name="image_input")
    list_encoder = [Conv2D(list_nb_filters[0], (3, 3),
                           strides=(2, 2), name="unet_conv2D_1", padding="same")(image_input)]    
    for i,f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
#        x = UpSampling2D(size=(2, 2))(x)
#        nb_filters = int(f / (2 ** (i + 1)))
        conv_encoded = conv_block_unet(list_encoder[-1], f, name, bn_axis)
        list_encoder.append(conv_encoded)   


    flat_model = Flatten(name='flatten_enc_GAN')(list_encoder[-1])
    
    fc6 = Dense(512, activation='relu', name='fc1_enc_GAN')(flat_model)
    bn_1 = BatchNormalization(name='1_bn')(fc6)
    fc7 = Dense(512, activation='relu', name='fc2_enc_GAN')(bn_1)
    bn_2 = BatchNormalization(name='2_bn')(fc7)
    
    output = Dense(n_class, activation='softmax', name='output_enc_GAN')(bn_2)      
    # Noise input and reshaping
   
#    conv_noise = Concatenate(axis=bn_axis)([list_encoder[-1],noise_input])

    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [up_conv_block_unet(list_encoder[-1],list_nb_filters[0], "unet_upconv2D_1",  bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], f, name, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(1, (3, 3), name="last_conv", padding="same")(x)
    x = Activation("tanh",name='generated_image_out')(x)

    generator_model = Model(inputs=[image_input], outputs=[x, output], name = model_name)
    generator_model.summary()
    return generator_model

