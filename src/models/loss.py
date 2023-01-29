# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

import tensorflow.keras.backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred):
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return tf.reduce_mean(1.0 - dice_coef(y_true, y_pred))


def jacard_coef(y_true, y_pred):
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth
    )


def jacard_coef_loss(y_true, y_pred):
    return tf.reduce_mean(1.0 - jacard_coef(y_true, y_pred))


def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(1.0 - tf.image.ssim(y_true, y_pred, 2.0))


def DISLoss(y_true, y_pred):
    return (
        dice_loss(y_true, y_pred)
        + jacard_coef_loss(y_true, y_pred)
        + ssim_loss(y_true, y_pred)
    )
