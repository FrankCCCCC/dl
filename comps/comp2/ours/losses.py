# Define Focal loss and smooth L1 loss

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Implement Focal Loss, done, pass test
def focal(alpha=0.25, gamma=2.0):
    '''
    Here we follow the paper "Focal Loss for Dense Object Detection" to implement Focal Loss
    The focal loss is used to compute the loss of the classification and it increase the weight of the positive samples to 
    enhance the performance to solve the inbalance negative samples problem.

    The formular:
    Hyperparameters: alpha, gamma
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where pt is:
    p_t = p, if y = 1, otherwise = 1 - p
    alpha_t is the weight of the cross entropy loss:
    alpha_t = alpha, if y = 1, otherwise = 1 - alpha

    The shape of the input:
    y_true = [batch size, number of anchors, number of classes + 1]
    y_pred = [batch size, number of anchors, number of classes]

    Note that the last element of dimension 3(number of classes + 1) is anchor state.
    It can represent the corresponding anchor prediction is an object, background, or ignore.
    Anchor State = -1 for ignore, 0 for background, 1 for object
    '''
    def _focal(y_true, y_pred):
        true_labels = y_true[:, :, :-1]
        true_anchor_states = y_true[:, :, -1]

        # Filter out "ignore" anchors
        true_not_ignore_indices = tf.where(tf.math.not_equal(true_anchor_states, -1))
        true_labels = tf.gather_nd(true_labels, true_not_ignore_indices)
        pred_labels = tf.gather_nd(y_pred, true_not_ignore_indices)

        # Compute Focal Loss
        alphas = tf.ones_like(true_labels) * alpha
        alpha_ts = tf.where(tf.equal(true_labels, 1), alphas, 1 - alphas)
        focal_weights = tf.where(tf.equal(true_labels, 1), 1 - pred_labels, pred_labels)
        focal_weights = alpha_ts * focal_weights ** gamma
        loss = focal_weights * keras.backend.binary_crossentropy(true_labels, pred_labels)

        # Normalize
        # epsilon = 1e-9
        normalizer = tf.where(tf.equal(true_anchor_states, 1))
        normalizer = keras.backend.cast(tf.keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        # return tf.reduce_sum(loss) / (normalizer + epsilon)
        return tf.reduce_sum(loss) / normalizer

    return _focal

# Implement smooth L1 Loss, done, pass test
def smooth_l1(sigma=3.0):
    sigma_sq = sigma ** 2
    def _smooth_l1(y_true, y_pred):
        true_regression = y_true[:, :, :-1]
        true_anchor_states = y_true[:, :, -1]

        # Filter out "ignore" anchors
        true_not_ignore_indices = tf.where(tf.math.equal(true_anchor_states, 1))
        pred_regression = tf.gather_nd(y_pred, true_not_ignore_indices)
        true_regression = tf.gather_nd(true_regression, true_not_ignore_indices)

        # Compute Smooth L1 Loss
        # L1 Loss = 0.5 * (sigma * x)^2     if |x| < 1 / (sigma^2)
        #           |x| - 0.5 / (sigma^2)   Otherwise
        x = tf.math.abs(pred_regression - true_regression)
        l1_loss = tf.where(
            tf.math.less(x, 1.0 / sigma_sq),
            0.5 * sigma_sq * tf.math.pow(x, 2),
            x - 0.5 / sigma_sq
        )

        # Normalize
        # epsilon = 1e-9
        normalizer = keras.backend.maximum(1, tf.keras.backend.shape(true_not_ignore_indices)[0])
        normalizer = keras.backend.cast(normalizer, keras.backend.floatx())

        # return tf.reduce_sum(l1_loss) / (normalizer + epsilon)
        return tf.reduce_sum(l1_loss) / normalizer

    return _smooth_l1