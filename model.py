import numpy as np
import json
import keras
from keras.models import *
from keras.layers import *
from types import MethodType
from train_and_predict import *

IMAGE_ORDERING = 'channels_last'

def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.evaluate_segmentation = MethodType(evaluate, model)

    return model


def vanilla_encoder(input_height=224,  input_width=224):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    
    img_input = Input(shape=(input_height, input_width, 3))

    x = img_input
    levels = []
    
    for n_filter in [64, 128, 256, 256, 256]:
        x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (Conv2D(n_filter, (kernel, kernel),
                    data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size),
             data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels

def segnet_decoder(f, n_classes):

    o = f
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    for n_filter in [256, 128, 64]:
        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (Conv2D(n_filter, (3, 3), padding='valid',data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    return o


def _segnet(n_classes, encoder,  input_height=416, input_width=608,
            encoder_level=3):

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width)

    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes)
    model = get_segmentation_model(img_input, o)

    return model


def segnet(n_classes, input_height=416, input_width=608, encoder_level=3):

    model = _segnet(n_classes, vanilla_encoder, input_height=input_height,
                    input_width=input_width, encoder_level=encoder_level)
    model.model_name = "segnet"
    return model
