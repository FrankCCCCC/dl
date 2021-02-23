# Test modules correct or not

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Customized modules
import model
from efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

# According to EfficientDet paper to set weighted BiFPN depth and depth of heads
weighted_bifpn = [64, 88, 112, 160, 224, 288, 384]
depth_bifpns = [3, 4, 5, 6, 7, 7, 8]
depth_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3]

def test_SeparableConvBlock_c():
    phi = 0
    img_inputs = keras.Input(shape=(image_sizes[phi]))
    x = model.SeparableConvBlock(num_channels=3, kernel_size=3, strides=1, name="eff_test")(img_inputs)
    outputs = model.SeparableConvBlock(num_channels=3, kernel_size=3, strides=1, name="eff_test")(x)
    test_model = keras.Model(inputs=img_inputs, outputs=outputs, name='test_SeparableConvBlock_c')
    # test_model()

if __name__ == '__main__':
    test_SeparableConvBlock_c()