#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D
from tensorflow.keras.applications.resnet50 import ResNet50


def Upsample(tensor, size):
    name = tensor.name.split('/')[0] + '_upsample'
    '''bilinear upsampling'''
    def bilinear_upsample(x, size):
        resized = tf.image.resize_bilinear(images=x, size=size, align_corners=True)
        return resized
    y = Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size, name=name)(tensor)
    return y


def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d')(y_pool)
    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_1')(tensor)
    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_6')(tensor)
    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_12')(tensor)
    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_18')(tensor)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')
    aspp_out = Upsample(tensor=y, size=[dims[1] * 4, dims[2] * 4])
    return aspp_out


def DeepLabV3Plus(img_height, img_width, nclasses=66):
    print('*** Building DeepLabv3Plus Network ***')

    base_model = ResNet50(input_shape=(
        img_height, img_height, 3), weights='imagenet', include_top=False)
    image_features = base_model.get_layer('activation_39').output
    x_a = ASPP(image_features)

    x_b = base_model.get_layer('add_2').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', name='low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)

    x = concatenate([x_a, x_b], name='decoder_concat')
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_1')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_2')(x)

    x = Conv2D(nclasses, (1, 1), padding='same', name='output_layer')(x)
    x = Upsample(x, [img_height, img_width])    
    x = Activation('softmax')(x)
    print('*** Building Network Completed***')    
    model = Model(inputs=base_model.input, outputs=x, name='DeepLabV3_Plus')
    print(f'*** Output_Shape => {model.output_shape} ***')    
    return model
