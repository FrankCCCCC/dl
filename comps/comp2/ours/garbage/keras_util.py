# Copy from Keras official code and modify it as backbone
# Customize TF Keras Modules

import functools
import cv2
import numpy as np

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None

# def inject_keras_modules(func):
#     import tensorflow.keras as keras
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         kwargs['backend'] = keras.backend
#         kwargs['layers'] = keras.layers
#         kwargs['models'] = keras.models
#         kwargs['utils'] = keras.utils
#         return func(*args, **kwargs)

#     return wrapper


def inject_tfkeras_modules(func):
    import tensorflow.keras as tfkeras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = tfkeras.backend
        kwargs['layers'] = tfkeras.layers
        kwargs['models'] = tfkeras.models
        kwargs['utils'] = tfkeras.utils
        return func(*args, **kwargs)

    return wrapper


# def init_keras_custom_objects():
#     import tensorflow.keras as keras
#     import efficientnet as model

#     custom_objects = {
#         'swish': inject_keras_modules(model.get_swish)(),
#         'FixedDropout': inject_keras_modules(model.get_dropout)()
#     }

#     keras.utils.generic_utils.get_custom_objects().update(custom_objects)


def init_tfkeras_custom_objects():
    import tensorflow.keras as tfkeras
    import efficientnet as model

    custom_objects = {
        'swish': inject_tfkeras_modules(model.get_swish)(),
        'FixedDropout': inject_tfkeras_modules(model.get_dropout)()
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)