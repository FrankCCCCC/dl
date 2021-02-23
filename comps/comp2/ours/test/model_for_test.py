from functools import reduce

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import initializers
from utils.utils import PriorProbability

# Customized modules
from efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from layers import wBiFPNAdd

# Hyperparameters
MOMENTUM = 0.997
EPSILON = 1e-4

# According to EfficientDet paper to set weighted BiFPN depth and depth of heads
# The backbones  of EfficientDet, done
'''
The corresponding backbone to EfficientDet Phi
B0 -> D0(phi 0), B1 -> D1(phi 1), B2 -> D2(phi 2), B3 -> D3(phi 3), B4 -> D4(phi 4), B5 -> D5(phi 5), B6 -> D6(phi 6)
B6 -> D7(phi 7), B7 -> D7X(phi 8) (IMPORTANT)
The value of phi is corresponding to the order of the following backbone list
'''
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB6, EfficientNetB7]

# The width of BiFPN which is the number of channels, also named 'fpn_num_filters' in efficientdet-tf2/efficientdet.py, done
# The formular of the paper is W = 64 * (1.35 ^ phi)
w_bifpns = [64, 88, 112, 160, 224, 288, 384, 384, 384]

# The depth of BiFPN which is the number of layers, also named 'fpn_cell_repeats' in efficientdet-tf2/efficientdet.py, done
# The formular of the paper is D = 3 + phi
d_bifpns = [3, 4, 5, 6, 7, 7, 8, 8, 8]

# The input image size of EfficientDet, done
'''
It is weired that from original paper, the input image size should be following
the input image size of EfficientDet of phi 6, 7, 8(7X) is 1280, 1536, 1536
image_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
'''
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408, 1408]

# The layers of BoxNet & ClassNet
depth_heads = [3, 3, 3, 4, 4, 4, 5, 5, 5]

# Reproduce original, done
def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True, name=f'{name}/conv')
    f2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

# Rewrite SeparableConvBlock with layer subclass, done
class SeparableConvBlock_c(keras.layers.Layer):
    def __init__(self, num_channels, kernel_size, strides, name, momentum=MOMENTUM, epsilon=EPSILON, freeze_bn=False):
        super(SeparableConvBlock_c, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.name = name
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        self.f1 = keras.layers.SeparableConv2D(
            self.num_channels, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same', use_bias=True, 
            name=f'{self.name}/conv2d')

        self.f2 = keras.layers.BatchNormalization(
            momentum=self.momentum, 
            epsilon=self.epsilon, 
            name=f'{self.name}/bn')
    
    def call(self, inputs):
        return self.f2(self.f1(inputs))

'''
According to the paper, for each BiFPN block, it use depthwise separable convolution for
feature fusion, and add batch normalization and activation after each convolution.
Process:
Depthwise separable convolution -> batch normalization -> activation
''' 
# Build Weighted BiFPN, done
def build_wBiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        '''
        For the first layer of BiFPN, we can only use P3, P4, P5 as inputs
        Don't know why, but many implementation do the same thing
        Depthwise separable convolution -> batch normalization
        '''
        pre = 'pre_'
        bifpn = 'bifpn-'
        _, _, C3, C4, C5 = features

        pn = 3
        P3_in = C3
        P3_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'{bifpn}{id}/{pn}/{pre}conv2d')(P3_in)
        P3_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'{bifpn}{id}/{pn}/{pre}bn')(P3_in)
        pn = 4
        P4_in = C4
        P4_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pre}conv2d')(P4_in)
        P4_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn')(P4_in)
        P4_in_2 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pn}/{pre}conv2d-2')(P4_in)
        P4_in_2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn-2')(P4_in_2)
        
        pn = 5
        P5_in = C5
        P5_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pn}/{pre}conv2d')(P5_in)
        P5_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn')(P5_in)
        P5_in_2 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pn}/{pre}conv2d-2')(P5_in)
        P5_in_2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn-2')(P5_in_2)

        pn = 6
        P6_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same', name=f'{bifpn}{id}/{pn}/{pre}conv2d')(C5)
        P6_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{bifpn}{id}/{pn}/{pre}bn')(P6_in)
        P6_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name=f'{bifpn}{id}/{pn}/{pre}maxpool')(P6_in)

        pn = 7
        P7_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name=f'{bifpn}{id}/{pn}/{pre}maxpool')(P6_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features

    # Top-Down & Bottom-Up BiFPN
    '''
    Px_Up: Top-down(upsampling implementation) to resize the feature map
    Px_Down: Bottom-Up convolution
    Px_td: The intermediate nodes of the layer
    Px_out: The output nodes of the layer

    illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
    refer by zylo117/Yet-Another-EfficientDet-Pytorch

    The process: 
    Depthwise separable convolution -> batch normalization and activation
    '''
    pre = 'fpn_top_down'
    bifpn = 'bifpn-'

    # Top-Down
    # Top-down upsampling
    P7_Up = keras.layers.UpSampling2D()(P7_in)

    pn = 6
    # Do Fast Normalized Fusion, then we get the intermediate nodes
    P6_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P6_in, P7_Up])
    '''
    Although the paper says "add batch normalization and activation after each convolution.", 
    in most of the implementation, they always apply swish before convolution
    '''
    # The original code for swish
    # P6_td = layers.Activation(tf.nn.swish)(P6_td)
    P6_td = tf.nn.swish(P6_td)
    # Separable Convolution and Batch Normalization
    P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P6_td)
    P6_Up = keras.layers.UpSampling2D()(P6_td)

    pn = 5
    # P5_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_1, P6_Up])
    P5_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in, P6_Up])
    # P5_td = layers.Activation(tf.nn.swish)(P5_td)
    P5_td = tf.nn.swish(P5_td)
    P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P5_td)
    P5_Up = keras.layers.UpSampling2D()(P5_td)

    pn = 4
    # P4_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_1, P5_Up])
    P4_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in, P5_Up])
    # P4_td = layers.Activation(tf.nn.swish)(P4_td)
    P4_td = tf.nn.swish(P4_td)
    P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                               name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P4_td)
    P4_Up = keras.layers.UpSampling2D()(P4_td) 

    pn = 3
    P3_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P3_in, P4_Up])
    # P3_out = layers.Activation(tf.nn.swish)(P3_out)
    P3_out = tf.nn.swish(P3_out)
    P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P3_out)

    # Bottom-Up
    pre = 'fpn_bottom_up'

    # Bottom-Up pooling
    P3_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)

    pn = 4
    if id == 0:
        P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_2, P4_td, P3_Down])
    else:
        P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in, P4_td, P3_Down])
    # P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_2, P4_td, P3_Down])
    # P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in, P4_td, P3_Down])
    # The original code for swish
    # P4_out = layers.Activation(tf.nn.swish)(P4_out)
    P4_out = tf.nn.swish(P4_out)
    P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P4_out)
    P4_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

    pn = 5
    if id == 0:
        P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_2, P5_td, P4_Down])
    else:
        P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in, P5_td, P4_Down])    
    # P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_2, P5_td, P4_Down])
    # P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in, P5_td, P4_Down])
    # P5_out = layers.Activation(tf.nn.swish)(P5_out) 
    P5_out = tf.nn.swish(P5_out)
    P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P5_out)
    P5_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)

    pn = 6
    P6_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P6_in, P6_td, P5_Down])
    # P6_out = layers.Activation(tf.nn.swish)(P6_out)
    P6_out = tf.nn.swish(P6_out)
    P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P6_out)
    P6_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)

    pn = 7
    P7_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P7_in, P6_Down])
    # P7_out = layers.Activation(tf.nn.swish)(P7_out)
    P7_out = tf.nn.swish(P7_out)
    P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P7_out)
    
    return [P3_out, P4_out, P5_out, P6_out, P7_out]

def build_wBiFPN_o(features, num_channels, id, freeze_bn=False):
    if id == 0:
        '''
        For the first layer of BiFPN, we can only use P3, P4, P5 as inputs
        Don't know why, but many implementation do the same thing
        Depthwise separable convolution -> batch normalization
        '''
        bifpn = 'bifpn-'
        pre = 'pre_'

        _, _, C3, C4, C5 = features

        pn = 3
        P3_in = C3
        P3_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'{bifpn}{id}/{pn}/{pre}conv2d')(P3_in)
        P3_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                          name=f'{bifpn}{id}/{pn}/{pre}bn')(P3_in)
        pn = 4
        P4_in = C4
        P4_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pre}conv2d')(P4_in)
        P4_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn')(P4_in)
        P4_in_2 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pn}/{pre}conv2d-2')(P4_in)
        P4_in_2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn-2')(P4_in_2)
        
        pn = 5
        P5_in = C5
        P5_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pn}/{pre}conv2d')(P5_in)
        P5_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn')(P5_in)
        P5_in_2 = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'{bifpn}{id}/{pn}/{pre}conv2d-2')(P5_in)
        P5_in_2 = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                            name=f'{bifpn}{id}/{pn}/{pre}bn-2')(P5_in_2)

        pn = 6
        P6_in = keras.layers.Conv2D(num_channels, kernel_size=1, padding='same', name=f'{bifpn}{id}/{pn}/{pre}conv2d')(C5)
        P6_in = keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{bifpn}{id}/{pn}/{pre}bn')(P6_in)
        P6_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name=f'{bifpn}{id}/{pn}/{pre}maxpool')(P6_in)

        pn = 7
        P7_in = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name=f'{bifpn}{id}/{pn}/{pre}maxpool')(P6_in)
        #-------------------------------------------------------------------------#

        # Top-Down & Bottom-Up BiFPN
        '''
        Px_Up: Top-down(upsampling implementation) to resize the feature map
        Px_Down: Bottom-Up convolution
        Px_td: The intermediate nodes of the layer
        Px_out: The output nodes of the layer

        illustration of a minimal bifpn unit
                P7_0 -------------------------> P7_2 -------->
                |-------------|                ↑
                                ↓                |
                P6_0 ---------> P6_1 ---------> P6_2 -------->
                |-------------|--------------↑ ↑
                                ↓                |
                P5_0 ---------> P5_1 ---------> P5_2 -------->
                |-------------|--------------↑ ↑
                                ↓                |
                P4_0 ---------> P4_1 ---------> P4_2 -------->
                |-------------|--------------↑ ↑
                                |--------------↓ |
                P3_0 -------------------------> P3_2 -------->
        refer by zylo117/Yet-Another-EfficientDet-Pytorch

        The process: 
        Depthwise separable convolution -> batch normalization and activation
        '''
        pre = 'fpn_top_down'
        bifpn = 'bifpn-'

        # Top-Down
        # Top-down upsampling
        P7_Up = keras.layers.UpSampling2D()(P7_in)

        pn = 6
        # Do Fast Normalized Fusion, then we get the intermediate nodes
        P6_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P6_in, P7_Up])
        '''
        Although the paper says "add batch normalization and activation after each convolution.", 
        in most of the implementation, they always apply swish before convolution
        '''
        # The original code for swish
        # P6_td = layers.Activation(tf.nn.swish)(P6_td)
        P6_td = tf.nn.swish(P6_td)
        # Separable Convolution and Batch Normalization
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P6_td)
        P6_Up = keras.layers.UpSampling2D()(P6_td)

        pn = 5
        # P5_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_1, P6_Up])
        P5_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in, P6_Up])
        # P5_td = layers.Activation(tf.nn.swish)(P5_td)
        P5_td = tf.nn.swish(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P5_td)
        P5_Up = keras.layers.UpSampling2D()(P5_td)

        pn = 4
        # P4_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_1, P5_Up])
        P4_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in, P5_Up])
        # P4_td = layers.Activation(tf.nn.swish)(P4_td)
        P4_td = tf.nn.swish(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P4_td)
        P4_Up = keras.layers.UpSampling2D()(P4_td) 

        # Bottom-Up
        pre = 'fpn_bottom_up'

        pn = 3
        P3_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P3_in, P4_Up])
        # P3_out = layers.Activation(tf.nn.swish)(P3_out)
        P3_out = tf.nn.swish(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P3_out)
        # Bottom-Up pooling
        P3_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)

        pn = 4
        P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_2, P4_td, P3_Down])
        # P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in, P4_td, P3_Down])
        # The original code for swish
        # P4_out = layers.Activation(tf.nn.swish)(P4_out)
        P4_out = tf.nn.swish(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P4_out)
        P4_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

        pn = 5
        P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_2, P5_td, P4_Down])
        # P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in, P5_td, P4_Down])
        # P5_out = layers.Activation(tf.nn.swish)(P5_out) 
        P5_out = tf.nn.swish(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P5_out)
        P5_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)

        pn = 6
        P6_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P6_in, P6_td, P5_Down])
        # P6_out = layers.Activation(tf.nn.swish)(P6_out)
        P6_out = tf.nn.swish(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P6_out)
        P6_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)

        pn = 7
        P7_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P7_in, P6_Down])
        # P7_out = layers.Activation(tf.nn.swish)(P7_out)
        P7_out = tf.nn.swish(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P7_out)

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        pre = 'fpn_top_down'
        bifpn = 'bifpn-'

        # Top-Down
        # Top-down upsampling
        P7_Up = keras.layers.UpSampling2D()(P7_in)

        pn = 6
        # Do Fast Normalized Fusion, then we get the intermediate nodes
        P6_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P6_in, P7_Up])
        '''
        Although the paper says "add batch normalization and activation after each convolution.", 
        in most of the implementation, they always apply swish before convolution
        '''
        # The original code for swish
        # P6_td = layers.Activation(tf.nn.swish)(P6_td)
        P6_td = tf.nn.swish(P6_td)
        # Separable Convolution and Batch Normalization
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P6_td)
        P6_Up = keras.layers.UpSampling2D()(P6_td)

        pn = 5
        # P5_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_1, P6_Up])
        P5_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in, P6_Up])
        # P5_td = layers.Activation(tf.nn.swish)(P5_td)
        P5_td = tf.nn.swish(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P5_td)
        P5_Up = keras.layers.UpSampling2D()(P5_td)

        pn = 4
        # P4_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_1, P5_Up])
        P4_td = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in, P5_Up])
        # P4_td = layers.Activation(tf.nn.swish)(P4_td)
        P4_td = tf.nn.swish(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P4_td)
        P4_Up = keras.layers.UpSampling2D()(P4_td) 

        pn = 3
        P3_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P3_in, P4_Up])
        # P3_out = layers.Activation(tf.nn.swish)(P3_out)
        P3_out = tf.nn.swish(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P3_out)

        # Bottom-Up
        pre = 'fpn_bottom_up'

        # Bottom-Up pooling
        P3_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)

        pn = 4
        P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in, P4_td, P3_Down])
        # P4_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P4_in_2, P4_td, P3_Down])
        # The original code for swish
        # P4_out = layers.Activation(tf.nn.swish)(P4_out)
        P4_out = tf.nn.swish(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P4_out)
        P4_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)

        pn = 5
        P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in, P5_td, P4_Down])    
        # P5_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P5_in_2, P5_td, P4_Down])
        # P5_out = layers.Activation(tf.nn.swish)(P5_out) 
        P5_out = tf.nn.swish(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P5_out)
        P5_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)

        pn = 6
        P6_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P6_in, P6_td, P5_Down])
        # P6_out = layers.Activation(tf.nn.swish)(P6_out)
        P6_out = tf.nn.swish(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P6_out)
        P6_Down = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)

        pn = 7
        P7_out = wBiFPNAdd(name=f'{bifpn}{id}/{pn}/{pre}wadd')([P7_in, P6_Down])
        # P7_out = layers.Activation(tf.nn.swish)(P7_out)
        P7_out = tf.nn.swish(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'{bifpn}{id}/{pn}/{pre}sepconv')(P7_out)

    return [P3_out, P4_out, P5_out, P6_out, P7_out]

# Implement ClassNet, done
class ClassNet(models.Model):
    def __init__(self, width, depth, num_classes=20, num_anchors=9, separable_conv=True, freeze_bn=False, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv

        if self.separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            self.convs = [layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                                 kernel_size=3, strides=1, padding='same',**kernel_initializer)
                          for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_classes * num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', kernel_size=3, strides=1, padding='same',
                                               **kernel_initializer)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            self.convs = [layers.Conv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                        kernel_size=3, strides=1, padding='same',**kernel_initializer)
                          for i in range(depth)]
            self.head = layers.Conv2D(filters=num_classes * num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', kernel_size=3, strides=1, padding='same',**kernel_initializer)
        self.bns = [
            [layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/class-{i}-bn-{j}') for j
             in range(3, 8)]
            for i in range(depth)]

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            feature = layers.Lambda(lambda x: tf.nn.swish(x))(feature)
        outputs = self.head(feature)
        outputs = layers.Reshape((-1, self.num_classes))(outputs)
        outputs = layers.Activation('sigmoid')(outputs)
        level += 1
        return outputs

# Implement BoxNet, done
class BoxNet(models.Model):
    def __init__(self, width, depth, num_anchors=9, separable_conv=True, freeze_bn=False, detect_quadrangle=False, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        self.detect_quadrangle = detect_quadrangle
        num_values = 9 if detect_quadrangle else 4

        if separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            self.convs = [layers.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', kernel_size=3, strides=1, padding='same', bias_initializer='zeros', **kernel_initializer) for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_anchors * num_values,name=f'{self.name}/box-predict', kernel_size=3, strides=1, padding='same', bias_initializer='zeros', **kernel_initializer)
        
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            self.convs = [layers.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', kernel_size=3, strides=1, padding='same', bias_initializer='zeros', **kernel_initializer) for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_anchors * num_values,name=f'{self.name}/box-predict', kernel_size=3, strides=1, padding='same', bias_initializer='zeros', **kernel_initializer)
        
        self.bns = [[layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/box-{i}-bn-{j}') for j in range(3, 8)]for i in range(depth)]
        self.reshape = layers.Reshape((-1, num_values))
        self.level = 0
    
    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
                feature = self.convs[i](feature)
                feature = self.bns[i][self.level](feature)
                feature = layers.Lambda(lambda x: tf.nn.swish(x))(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        # outputs = layers.Activation('sigmoid')(outputs)
        self.level += 1
        return outputs
    

def get_efficientdet_info(phi):
    return {
        'BiFPN_width': w_bifpns[phi],
        'BiFPN_depth': d_bifpns[phi],
        'image_size': image_sizes[phi],
        'image_shape': (image_sizes[phi], image_sizes[phi]),
        'depth_head': depth_heads[phi],
    }

# Implement EfficientDet, done
def EfficientDet(phi, num_classes = 20, num_anchors = 9, freeze_bn=False):
    # Phi 0(D0) ~ 8(7X)
    assert(phi < 9)
    width_bifpn = w_bifpns[phi]
    depth_bifpn = d_bifpns[phi]
    depth_head = depth_heads[phi]
    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)
    backbone = backbones[phi]
    
    img_inputs = keras.layers.Input(shape=input_shape)
    features2bifpn = backbone(input_tensor=img_inputs, freeze_bn=freeze_bn)
    
    # Here we only implement weighted BiFPN becasue of better performance
    for i in range(depth_bifpn):
        features2bifpn = build_wBiFPN_o(features2bifpn, width_bifpn, i)
    bifpn_out = features2bifpn

    # Class Net
    class_net = ClassNet(width_bifpn, depth_head, num_classes=num_classes, freeze_bn=freeze_bn, name='class_net')
    classifier = [class_net.call([feature, i]) for i, feature in enumerate(bifpn_out)]
    classifier_out = layers.Concatenate(axis=1, name="classifier")(classifier)

    # Box Net
    box_net = BoxNet(width_bifpn, depth_head, num_anchors=num_anchors, freeze_bn=freeze_bn,name="box_net")
    regressor = [box_net.call([feature, i]) for i, feature in enumerate(bifpn_out)]
    regressor_out = layers.Concatenate(axis=1, name="regressor")(regressor)

    model = keras.models.Model(inputs=[img_inputs], outputs=[regressor_out, classifier_out], name='efficientdet')

    return model