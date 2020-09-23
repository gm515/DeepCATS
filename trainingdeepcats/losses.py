"""
Loss Functions
Author: Gerald M

Loss functions to use for training. Some are adapted from Lar's Blog,
https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
"""

from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

epsilon = 1e-6
smooth = 1


def binary_focal_loss(y_true, y_pred):
    gamma=2.
    alpha=.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    # return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def focal_loss(y_true, y_pred):
    alpha=0.25
    gamma=2
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    # return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss


# Combined binary cross entropy and dice loss
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + K.log(dsc(y_true, y_pred))
    return loss


# Combo loss from https://github.com/asgsaeid/ComboLoss
def combo_loss(y_true, y_pred):
    '''
    ce_w values smaller than 0.5 penalize false positives more while values larger than 0.5 penalize false negatives more
    ce_d_w is level of contribution of the cross-entropy loss in the total loss.
    '''

    ce_w = 0.4
    ce_d_w = 0.5
    e = K.epsilon()
    smooth = 1

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    d = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    y_pred_f = K.clip(y_pred_f, e, 1.0 - e)
    out = - (ce_w * y_true_f * K.log(y_pred_f)) + ((1 - ce_w) * (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_d_w * weighted_ce) - ((1 - ce_d_w) * (1 - d))

    return combo


def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall


def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
    return tp


def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.4#0.7#0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)


def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


def bce_focal_tversky_loss(alpha):
    def loss(y_true, y_pred):
        return ((alpha*binary_crossentropy(y_true, y_pred)) + ((1-alpha)*focal_tversky(y_true,y_pred)))
    return loss


def dice_focal_tversky_loss(alpha):
    def loss(y_true, y_pred):
        return ((alpha*dice_loss(y_true, y_pred)) + ((2-alpha)*focal_tversky(y_true,y_pred)))
    return loss


def dice_surface_loss(y_true, y_pred):
    loss = dice_loss(y_true, y_pred) + 0.5*surface_loss(y_true,y_pred)
    return loss


def bce_surface_loss(alpha):
    def combine_loss(y_true, y_pred):
        return alpha * binary_crossentropy(y_true, y_pred) + (1 - alpha) * surface_loss(y_true, y_pred)
    return combine_loss


def dsc_surface_loss(alpha):
    def combine_loss(y_true, y_pred):
        return alpha * dsc(y_true, y_pred) + (1 - alpha) * surface_loss(y_true, y_pred)
    return combine_loss


def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.math.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

  return loss


# beta = 0.01 means negative class is 1% of the total samples so it is weighted in the loss accordingly
def balanced_cross_entropy(beta):
    def convert_to_logits(y_pred):
      y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

      return tf.math.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss


def iou(y_true, y_pred):
    def f1(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        return 1 - K.mean( (intersection + epsilon) / (union + epsilon) )
    def f2(y_true, y_pred):
        y_true = 1-y_true
        y_pred = 1-y_pred
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        return 1 - K.mean( (intersection + epsilon) / (union + epsilon) )

    return tf.cond(tf.equal(K.max(y_true), 0), lambda: f1(y_true, y_pred), lambda: f2(y_true, y_pred))


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)
