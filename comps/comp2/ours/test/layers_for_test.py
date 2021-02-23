# Define customized layers
import tensorflow as tf
from tensorflow import keras

# Useless
# Identical to keras.layers.BatchNormalization, but add control to the momentum and epsilon trainable or not
# class BatchNormalization(keras.layers.BatchNormalization):
#     def __init__(self, *args, **kwargs):
#         super(BatchNormalization, self).__init__(*args, **kwargs)

#     def call(self, inputs, training=None, **kwargs):
#         # training is a call argument, when model is training, it would be True value and vice versa
#         # Official: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
#         # BatchNormalize Intro: https://stackoverflow.com/questions/50047653/set-training-false-of-tf-layers-batch-normalization-when-training-will-get-a
#         # When training is True, it would update moving average of mean and variance. Thus, we can get a better normalization result
#         # If the self.trainable is set to True, it would let training be true
#         if not training:
#             return super(BatchNormalization, self).call(inputs, training=False, **kwargs)
#         else:
#             return super(BatchNormalization, self).call(inputs, training=(not self.trainable), **kwargs)

# Implement "Fast Normalized Fusion", done
class wBiFPNAdd(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        input_dim = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                shape=input_dim,
                                initializer=keras.initializers.Constant(1 / input_dim),
                                trainable=True,
                                dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.nn.relu(self.w)
        s = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = s / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config