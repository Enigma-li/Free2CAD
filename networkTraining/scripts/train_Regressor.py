#
# Project Free2CAD
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2022. All Rights Reserved.
#
# ==============================================================================
"""Regressor training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import time
import datetime
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import default
from loader import GPRegTFReader
from network import Regressor, AutoencoderEmbed, GpTransformer
from tensorflow.keras.preprocessing import image
import numpy as np

# Hyper Parameters
hyper_params = {
    'maxIter': 1500000,
    'batchSize': 16,
    'dbDir': '',
    'outDir': '',
    'device': '',
    'dispLossStep': 100,
    'exeValStep': 5000,
    'saveModelStep': 10000,
    'ckpt': '',
    'cnt': False,
    'rootSize': 16,
    'status:': 'train',
    'embedS_ckpt': '',
    'nb_layers': 6,
    'd_model': 256,
    'd_ff': 2048,
    'nb_heads': 8,
    'drop_rate': 0.1,
    'nb_stroke_max': 1024,
    'nb_gp_max': 64,
    'imgSize': 256,
    'maxS': 3,
}

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dbDir', required=True, help='TFRecords dataset directory', type=str)
parser.add_argument('--outDir', required=True, help='output directory', type=str)
parser.add_argument('--embedS_ckpt', required=True, help='stroke embedding checkpoint', type=str)
parser.add_argument('--gpTF_ckpt', required=True, help='grouping transformer checkpoint', type=str)
parser.add_argument('--devices', required=True, help='GPU device indices', type=str)
parser.add_argument('--ckpt', help='checkpoint path', type=str, default='')
parser.add_argument('--cnt', help='continue training flag', type=bool, default=False)
parser.add_argument('--status', help='training or testing flag', type=str, default='train')
parser.add_argument('--bSize', help='batch size', type=int, default=16)
parser.add_argument('--maxS', help='maxinum sliding window width', type=int, default=3)

args = parser.parse_args()
hyper_params['dbDir'] = args.dbDir
hyper_params['outDir'] = args.outDir
hyper_params['device'] = args.devices
hyper_params['ckpt'] = args.ckpt
hyper_params['cnt'] = args.cnt
hyper_params['status'] = args.status
hyper_params['embedS_ckpt'] = args.embedS_ckpt
hyper_params['gpTF_ckpt'] = args.gpTF_ckpt
hyper_params['batchSize'] = args.bSize
hyper_params['maxS'] = args.maxS

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = hyper_params['device']
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def create_padding_mask(seq, token):
    seq = tf.reduce_sum(seq, axis=2)
    seq = tf.cast(tf.math.equal(seq, token * hyper_params['d_model']), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp, -2.0)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp, -2.0)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar, -2.0)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


@tf.function
def loss_fn(gt_lineSeg, pred_lineSeg, full_stroke_map, gt_faceMap, pred_faceMap):
    
    # 1. face map prediction: l2
    face_mask = tf.cast(tf.math.logical_not(tf.math.equal(gt_faceMap, -1.0)), tf.float32)  # [N, nb_gp, 256, 256, 1]
    gt_faceMap = gt_faceMap * face_mask
    pred_faceMap = pred_faceMap * face_mask
    face_loss = tf.reduce_sum(tf.pow(gt_faceMap-pred_faceMap, 2))
    face_nb_elem = tf.reduce_sum(face_mask)
    face_loss_val = face_loss / face_nb_elem

    # 2. base line segmentation: l2
    bseg_mask = tf.cast(tf.math.logical_not(tf.math.equal(gt_lineSeg, -1.0)), tf.float32)  # [N, nb_gp, 256, 256, 1]
    base_img_shape = tf.shape(pred_lineSeg)
    Nb = base_img_shape[0]
    nb_gp = base_img_shape[1]
    fullOne = tf.ones([Nb, nb_gp, 256, 256, 1])
    fullStroke = tf.repeat(tf.expand_dims(full_stroke_map, axis=1), nb_gp, axis=1)  # [N, 256, 256, 1] -> [N, nb_gp, 256, 256, 1]
    stroke_mask = (fullOne - fullStroke) * bseg_mask
    pred_lineSeg = pred_lineSeg * stroke_mask
    gt_lineSeg = gt_lineSeg * stroke_mask
    bseg_loss = tf.reduce_sum(tf.pow(gt_lineSeg - pred_lineSeg, 2))
    bseg_elem = tf.reduce_sum(stroke_mask)
    bseg_loss_val = bseg_loss / bseg_elem

    total_loss = face_loss_val + bseg_loss_val

    return total_loss, face_loss_val, bseg_loss_val, pred_lineSeg, pred_faceMap


# define model
# Stroke embedding
modelAESSG = AutoencoderEmbed(code_size=hyper_params['d_model'], x_dim=hyper_params['imgSize'],
                              y_dim=hyper_params['imgSize'], root_feature=32)
embS_ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG)
embS_ckpt_manager = tf.train.CheckpointManager(embS_ckpt, hyper_params['embedS_ckpt'], max_to_keep=30)
if embS_ckpt_manager.latest_checkpoint:
    embS_ckpt.restore(embS_ckpt_manager.latest_checkpoint)
    print('restore stroke network from the checkpoint {}'.format(embS_ckpt_manager.latest_checkpoint))

# Grouping transformer
transformer = GpTransformer(num_layers=hyper_params['nb_layers'], d_model=hyper_params['d_model'],
                            num_heads=hyper_params['nb_heads'], dff=hyper_params['d_ff'],
                            pe_input=hyper_params['nb_stroke_max'], pe_target=hyper_params['nb_gp_max'],
                            rate=hyper_params['drop_rate'])
gp_ckpt = tf.train.Checkpoint(transformer=transformer)
gp_ckpt_manager = tf.train.CheckpointManager(gp_ckpt, hyper_params['gpTF_ckpt'], max_to_keep=30)
if gp_ckpt_manager.latest_checkpoint:
    gp_ckpt.restore(gp_ckpt_manager.latest_checkpoint)
    print('restore grouping network from the checkpoint {}'.format(gp_ckpt_manager.latest_checkpoint))

# regressor
regressor = Regressor(root_feature=hyper_params['rootSize'])


# define reader
readerTrain = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=hyper_params['batchSize'], shuffle=True,
                           raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=True, prefix='train')
readerEval = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=hyper_params['batchSize'], shuffle=False,
                          raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=False, prefix='eval')

# Optimizer
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

val_total_loss_metric = tf.keras.metrics.Mean(name='Validate_total_loss', dtype=tf.float32)
val_baseSeg_loss_metric = tf.keras.metrics.Mean(name='Validate_baseSegL2_loss', dtype=tf.float32)
val_faceSeg_loss_metric = tf.keras.metrics.Mean(name='Validate_faceSegL2_loss', dtype=tf.float32)
train_total_loss_metric = tf.keras.metrics.Mean(name='Train_total_loss', dtype=tf.float32)
train_baseSeg_loss_metric = tf.keras.metrics.Mean(name='Train_baseSegL2_loss', dtype=tf.float32)
train_faceSeg_loss_metric = tf.keras.metrics.Mean(name='Train_faceSegL2_loss', dtype=tf.float32)


@tf.function
def stroke_selection(group_output, full_stroke_input, gp_label, inp, tar):
    # group_output: [N, nb_g, nb_s]
    # full_stroke_input: [N, 256, 256, nb_s]

    # mask
    mask = tf.cast(tf.math.logical_not(tf.math.equal(gp_label, -1.0)), tf.float32)

    # calculate the real nb_g, nb_s
    # inp: [N, nb_s, 256]
    inp_sum = tf.reduce_sum(inp, axis=2)
    inp_sum_mask = tf.cast(tf.logical_not(tf.math.equal(inp_sum, -2.0 * 256)), tf.int32)  # [N, nb_s]
    nb_strokes = tf.reduce_sum(inp_sum_mask, axis=1)  # [N]

    # tar: [N, nb_g, 256]
    tar_sum = tf.reduce_sum(tar, axis=2)
    tar_sum_mask = tf.cast(tf.logical_not(tf.math.equal(tar_sum, -2.0 * 256)), tf.int32)  # [N, nb_g]
    nb_gps = tf.reduce_sum(tar_sum_mask, axis=1)  # [N]

    bSize = tf.shape(full_stroke_input)[0]
    group_output = tf.sigmoid(group_output) * mask  # soft-blend
    target_gp_nb = tf.shape(group_output)[1]

    gp_stroke_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for itr in range(bSize):
        cur_nb_s = nb_strokes[itr]
        cur_nb_g = nb_gps[itr]
        cur_strokes = tf.slice(full_stroke_input, [itr, 0, 0, 0], [1, -1, -1, cur_nb_s])  # [1, 256, 256, nb_s]
        cur_strokes = tf.transpose(cur_strokes, [3, 1, 2, 0])  # [nb_s, 256, 256, 1]
        cur_strokes = tf.subtract(1.0, cur_strokes)  # inverse stroke value: background-0.0, strokes-1.0
        cur_gp_labels = tf.slice(group_output, [itr, 0, 0], [1, cur_nb_g, cur_nb_s])  # [1, nb_g, nb_s]
        cur_gp_labels = tf.reshape(cur_gp_labels, [cur_nb_g, cur_nb_s])  # [nb_g, nb_s]

        cur_stroke_rep = tf.repeat(tf.expand_dims(cur_strokes, axis=0), cur_nb_g, axis=0)  # [nb_g, nb_s, 256, 256, 1]
        cur_gp_label_rep = tf.repeat(tf.expand_dims(cur_gp_labels, axis=2), 256, axis=2)  # [nb_g, nb_s, 256]
        cur_gp_label_rep = tf.repeat(tf.expand_dims(cur_gp_label_rep, axis=3), 256,
                                     axis=3)  # [nb_g, nb_s, 256, 256]
        cur_gp_label_rep = tf.reshape(cur_gp_label_rep,
                                      [cur_nb_g, cur_nb_s, 256, 256, 1])  # [nb_g, nb_s, 256, 256, 1]
        gp_strokes_sel = cur_gp_label_rep * cur_stroke_rep  # [nb_g, nb_s, 256, 256, 1]
        gp_stroke_max = tf.reduce_max(gp_strokes_sel, axis=1)  # [nb_g, 256, 256, 1]
        gp_strokes = tf.subtract(1.0, gp_stroke_max)  # convert stroke value back

        # padding
        gp_strokes_shape = tf.shape(gp_strokes)
        gp_strokes_pad = tf.pad(gp_strokes, [[0, target_gp_nb - gp_strokes_shape[0]],
                                             [0, 0], [0, 0], [0, 0]], constant_values=0.0)
        gp_stroke_list = gp_stroke_list.write(itr, gp_strokes_pad)

    gp_stroke_img = gp_stroke_list.stack()  # [N, 4, 256, 256, 1]

    return gp_stroke_img


@tf.function
def grouper_forward(inp, tar, allStroke_map, gp_label):
    
    # grouping
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar)
    # [N, nb_gp, nb_s]
    gp_prediction, _ = transformer(inp, tar, False, enc_padding_mask, combined_mask, dec_padding_mask)
    gp_stroke_img = stroke_selection(gp_prediction, allStroke_map, gp_label, inp, tar)

    return gp_stroke_img


train_step_signature = [
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256), dtype=tf.float32),  # input
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256), dtype=tf.float32),  # grouping token
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256, 256, 1), dtype=tf.float32),  # gt base segmentation
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256, 256, 1), dtype=tf.float32),  # gt face segmentation
    tf.TensorSpec(shape=(hyper_params['batchSize'], 256, 256, None), dtype=tf.float32),  # all strokes
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256, 256, 4), dtype=tf.float32),  # raw depth, normal maps
    tf.TensorSpec(shape=(hyper_params['batchSize'], 256, 256, 1), dtype=tf.float32), # full stroke image
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, None), dtype=tf.float32),  # gp label
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, gtLine_seg, gtFace_seg, allStroke_map, group_depthNormal_maps, full_stroke_map, gp_label):
    
    # input stroke image - soft-blend
    gp_stroke_img = grouper_forward(inp, tar, allStroke_map, gp_label)
 
    with tf.GradientTape() as tape:
        pred_faceMap, pred_baseMap = regressor(gp_stroke_img, group_depthNormal_maps, True)

        total_loss_val, faceSeg_loss_val, \
        baseSeg_loss_val, pred_lineSeg_masked, pred_faceSeg_masked = loss_fn(gtLine_seg, pred_baseMap,
                                                                             full_stroke_map,
                                                                             gtFace_seg, pred_faceMap)

    gradients = tape.gradient(total_loss_val, regressor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, regressor.trainable_variables))

    train_total_loss_metric.update_state(total_loss_val)
    train_baseSeg_loss_metric.update_state(baseSeg_loss_val)
    train_faceSeg_loss_metric.update_state(faceSeg_loss_val)

    return total_loss_val, pred_faceSeg_masked, pred_lineSeg_masked, gp_stroke_img


eval_step_signature = [
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256), dtype=tf.float32),  # input
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256), dtype=tf.float32),  # grouping token
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256, 256, 1), dtype=tf.float32),  # gt base segmentation
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256, 256, 1), dtype=tf.float32),  # gt face segmentation
    tf.TensorSpec(shape=(hyper_params['batchSize'], 256, 256, None), dtype=tf.float32),  # all strokes
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256, 256, 4), dtype=tf.float32),  # raw depth, normal maps
    tf.TensorSpec(shape=(hyper_params['batchSize'], 256, 256, 1), dtype=tf.float32), # full stroke image
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, None), dtype=tf.float32), # gp label
]


@tf.function(input_signature=eval_step_signature)
def eval_step(inp, tar, gtLine_seg, gtFace_seg, allStroke_map, group_depthNormal_maps, full_stroke_map, gp_label):
    
    # input stroke image - soft-blend
    gp_stroke_img = grouper_forward(inp, tar, allStroke_map, gp_label)

    # network forward
    pred_faceMap, pred_baseMap = regressor(gp_stroke_img, group_depthNormal_maps, False)

    total_loss_val, faceSeg_loss_val, \
    baseSeg_loss_val, pred_lineSeg_masked, pred_faceMap_masked = loss_fn(gtLine_seg, pred_baseMap,
                                                                         full_stroke_map,
                                                                         gtFace_seg, pred_faceMap)

    val_total_loss_metric.update_state(total_loss_val)
    val_baseSeg_loss_metric.update_state(baseSeg_loss_val)
    val_faceSeg_loss_metric.update_state(faceSeg_loss_val)

    return total_loss_val, pred_faceMap_masked, pred_lineSeg_masked, gp_stroke_img


# with K=maxS
def cook_raw(stroke_net, input_raw, glabel_raw, gmap_raw, nb_gps, nb_strokes, gp_embSize, global_nd, gbseg_raw, pregp_raw, startidx_raw, gfaceSeg_raw):
    gp_start_token = tf.fill([1, gp_embSize], -1.0)  # [1, 256]

    nb_batch = tf.shape(input_raw)[0]

    target_stroke_nb = tf.shape(input_raw)[3]
    # target_gp_nb = tf.shape(glabel_raw)[1]
    target_gp_nb = hyper_params['maxS']
    assert (target_stroke_nb == tf.shape(glabel_raw)[2])

    input_cook_list = []
    gp_token_cook_list = []
    label_cook_list = []
    gp_fSeg_cook_list = []
    full_stroke_list = []
    gp_stroke_list = []
    gp_depthNormal_cook_list = []
    gp_bSeg_cook_list = []
    for itr in range(nb_batch):
        # get group and stroke numbers
        nb_gp = min(nb_gps[itr], hyper_params['maxS'])
        nb_stroke = nb_strokes[itr]
        start_idx = tf.reshape(tf.slice(startidx_raw, [itr, 0], [1, 1]), [])  # []

        # ------ Get slice data ---------
        cur_input = tf.slice(input_raw, [itr, 0, 0, 0], [1, -1, -1, nb_stroke])  # [1, 256, 256, nb_stroke]
        cur_gp_label = tf.slice(glabel_raw, [itr, 0, 0], [1, nb_gp, nb_stroke])  # [1, nb_g, nb_s]
        cur_gp_label = tf.reshape(cur_gp_label, [nb_gp, nb_stroke])  # [nb_gp, nb_stroke]
        cur_gp_maps = tf.slice(gmap_raw, [itr, 0, 0, 0, 0], [1, nb_gp, -1, -1, -1])  # [1, nb_gp, 256, 256, 5]
        cur_gp_maps = tf.reshape(cur_gp_maps, [nb_gp, 256, 256, 5])  # [nb_g, 256, 256, 5]
        cur_gp_depthNormal_maps = tf.slice(cur_gp_maps, [0, 0, 0, 1], [-1, -1, -1, 4])  # [nb_g, 256, 256, 4]
        cur_global_gp_depthNormal_maps = tf.slice(global_nd, [itr, 0, 0, 0], [1, -1, -1, -1])  # [1, 256, 256, 4]
        cur_gp_bseg = tf.slice(gbseg_raw, [itr, 0, 0, 0, 0], [1, nb_gp, -1, -1, -1])  # [1, nb_gp, 256, 256, 1]
        cur_pregp = tf.reshape(tf.slice(pregp_raw, [itr, 0, 0], [1, -1, -1]), [1, gp_embSize, gp_embSize, 1])  # [1, 256, 256, 1]
        cur_gp_fseg = tf.slice(gfaceSeg_raw, [itr, 0, 0, 0, 0], [1, nb_gp, -1, -1, -1])  # [1, nb_gp, 256, 256, 1]

        # -------- Assemble data ------------

        # full stroke input
        cur_full_stroke = tf.reshape(tf.reduce_min(cur_input, axis=3), [1, 256, 256, 1])  # [1, 256, 256, 1]
        full_stroke_list.append(cur_full_stroke)

        # stroke embedding
        input_trans = tf.transpose(cur_input, [3, 1, 2, 0])  # [nb_stroke, 256, 256, 1]
        input_cook = stroke_net.encoder(input_trans, training=False)  # [nb_stroke, 256]

        # group token
        # [nb_g, nb_s, 256, 256, 1], repeat the first channel
        input_trans_rep = tf.repeat(tf.expand_dims(input_trans, axis=0), nb_gp, axis=0)
        # [nb_g, nb_s, 256, 256, 1], repeat the last three channels
        label_rep = tf.repeat(tf.expand_dims(cur_gp_label, axis=2), gp_embSize, axis=2)
        label_rep = tf.repeat(tf.expand_dims(label_rep, axis=3), gp_embSize, axis=3)
        label_rep = tf.reshape(label_rep, [nb_gp, nb_stroke, gp_embSize, gp_embSize, 1])  # [nb_g, nb_s, 256, 256, 1]
        gp_strokes_sel = label_rep * input_trans_rep  # [nb_g, nb_s, 256, 256, 1]
        gp_stroke_sum = tf.reduce_sum(gp_strokes_sel, axis=1)  # [nb_g, 256, 256, 1]
        label_rep_sum = tf.reduce_sum(label_rep, axis=1)  # [nb_g, 256, 256, 1]
        gp_strokes = tf.where(tf.math.equal(gp_stroke_sum, label_rep_sum), 1.0, 0.0)  # [nb_g, 256, 256, 1]
        gp_embed = stroke_net.encoder(gp_strokes, training=False)  # [nb_g, 256]

        # Add start token
        if start_idx > 0:
            gp_start_token = stroke_net.encoder(cur_pregp, training=False)  # [1, 256]
        gp_cook = tf.concat([gp_start_token, gp_embed], axis=0)  # [nb_gp+1, 256]

        # label: add end group label (all zeros)
        label_cook = tf.concat([cur_gp_label, tf.fill([1, nb_stroke], 0.0)], axis=0)

        # face map
        gp_fseg_cook = tf.reshape(cur_gp_fseg, [nb_gp, 256, 256, 1]) # [nb_gp, 256, 256, 1]

        # base line segmentation
        gp_bseg_cook = tf.reshape(cur_gp_bseg, [nb_gp, 256, 256, 1])  # [nb_gp, 256, 256, 1]
        gp_bseg_one = tf.ones([nb_gp, 256, 256, 1])
        gp_bseg_cook = gp_bseg_one - gp_bseg_cook       # flip to black bg, white line
		
        # depth, normal maps
        cur_gp_depthNormal_cook = tf.concat([cur_global_gp_depthNormal_maps,
                                             cur_gp_depthNormal_maps], axis=0)  # [nb_gp+1, 256, 256, 4]

        # ------- Padding --------
        # stroke input
        input_cook_shape = tf.shape(input_cook)  # [nb_stroke, 256]
        input_cook_pad = tf.pad(input_cook, [[0, target_stroke_nb - input_cook_shape[0]], [0, 0]],
                                constant_values=-2.0)  # [max_nb_sroke, 256]
        input_cook_pad = tf.reshape(input_cook_pad, [1, -1, input_cook_shape[1]])  # [1, max_nb_s, 256]
        input_cook_list.append(input_cook_pad)

        # gp token
        gp_cook_shape = tf.shape(gp_cook)  # [nb_gp+1, 256]
        gp_cook_pad = tf.pad(gp_cook, [[0, target_gp_nb - gp_cook_shape[0] + 1], [0, 0]],
                             constant_values=-2.0)  # [4, 256]
        gp_cook_pad = tf.reshape(gp_cook_pad, [1, -1, gp_cook_shape[1]])  # [1, 4, 256]
        gp_token_cook_list.append(gp_cook_pad)

        # face map
        gp_fseg_cook_shape = tf.shape(gp_fseg_cook)  # [nb_gp, 256, 256, 1]
        gp_fseg_cook_pad = tf.pad(gp_fseg_cook, [[0, target_gp_nb - gp_fseg_cook_shape[0] + 1],
                                                             [0, 0], [0, 0], [0, 0]], constant_values=-1.0)  # [4, 256, 256, 1]
        gp_fseg_cook_pad = tf.reshape(gp_fseg_cook_pad, [1, -1, 256, 256, 1])  # [1, 4, 256, 256, 1]
        gp_fSeg_cook_list.append(gp_fseg_cook_pad)

        # bSeg map
        gp_bseg_cook_shape = tf.shape(gp_bseg_cook)  # [nb_pg, 256, 256, 1]
        gp_bseg_cook_pad = tf.pad(gp_bseg_cook, [[0, target_gp_nb-gp_bseg_cook_shape[0] + 1],
                                                 [0, 0], [0, 0], [0, 0]], constant_values=-1.0)  # [4, 256, 256, 1]
        gp_bseg_cook_pad = tf.reshape(gp_bseg_cook_pad, [1, -1, 256, 256, 1])  # [1, 4, 256, 256, 1]
        gp_bSeg_cook_list.append(gp_bseg_cook_pad)

        # depth, normal maps
        cur_gp_depthNormal_cook_shape = tf.shape(cur_gp_depthNormal_cook)  # [nb_gp+1, 256, 256, 4]
        cur_gp_depthNormal_cool_pad = tf.pad(cur_gp_depthNormal_cook,
                                             [[0, target_gp_nb - cur_gp_depthNormal_cook_shape[0] + 1],
                                              [0, 0], [0, 0], [0, 0]], constant_values=-1.0)  # [4, 256, 256, 4]
        cur_gp_depthNormal_cool_pad = tf.reshape(cur_gp_depthNormal_cool_pad, [1, -1, 256, 256, 4])  # [1, 4, 256, 256, 4]
        gp_depthNormal_cook_list.append(cur_gp_depthNormal_cool_pad)

        # gp_strokes
        gp_strokes_shape = tf.shape(gp_strokes)  # [nb_g, 256, 256, 1]
        gp_strokes_pad = tf.pad(gp_strokes, [[0, target_gp_nb - gp_strokes_shape[0] + 1],
                                             [0, 0], [0, 0], [0, 0]], constant_values=0.0)  # [4, 256, 256, 1]
        gp_stroke_pad = tf.reshape(gp_strokes_pad, [1, -1, 256, 256, 1])  # [1, 4, 256, 256, 1]
        gp_stroke_list.append(gp_stroke_pad)

        # gp label
        label_cook_shape = tf.shape(label_cook)
        label_cook_pad = tf.pad(label_cook,
                                [[0, target_gp_nb - label_cook_shape[0] + 1],
                                 [0, target_stroke_nb - label_cook_shape[1]]], constant_values=-1.0)
        label_cook_pad = tf.reshape(label_cook_pad, [1, -1, target_stroke_nb])
        label_cook_list.append(label_cook_pad)

    # concat cook list after padding
    input_img = tf.concat(input_cook_list, axis=0)
    gp_token = tf.concat(gp_token_cook_list, axis=0)
    full_stroke_map = tf.concat(full_stroke_list, axis=0)
    gp_depthNormals = tf.concat(gp_depthNormal_cook_list, axis=0)
    gp_stroke_imgs = tf.concat(gp_stroke_list, axis=0)
    gp_bSeg_maps = tf.concat(gp_bSeg_cook_list, axis=0)
    gp_fSeg_maps = tf.concat(gp_fSeg_cook_list, axis=0)
    gp_label = tf.concat(label_cook_list, axis=0)

    return input_img, gp_token, full_stroke_map, gp_depthNormals, gp_stroke_imgs, gp_bSeg_maps, gp_fSeg_maps, gp_label


def train_net():
    # Set logging
    train_logger = logging.getLogger('main.training')
    train_logger.info('---Begin training: ---')

    # TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = output_folder + '/summary/train_' + current_time
    test_log_dir = output_folder + '/summary/test_' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Checkpoint
    if not hyper_params['ckpt']:
        hyper_params['ckpt'] = output_folder + '/checkpoint'
    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG, transformer=transformer, regressor=regressor, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=100)

    if hyper_params['cnt']:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            train_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    # Training process
    for step in range(hyper_params['maxIter']):

        # train step
        input_raw, glabel_raw, gmap_raw, _, _, _, _, global_nd, gbseg_raw, \
        nb_gps, nb_strokes, pregp_raw, startidx_raw, gfaceSeg_raw = readerTrain.next()

        input_img, gp_token, full_stroke_map, group_depthNormals, \
        gt_gp_strokes, gt_bSeg_map, gt_faceSeg_map, gp_label = cook_raw(modelAESSG, input_raw, glabel_raw,
                                                                        gmap_raw, nb_gps, nb_strokes,
                                                                        hyper_params['d_model'], global_nd, gbseg_raw,
                                                                        pregp_raw, startidx_raw, gfaceSeg_raw)

        train_total_loss_val, pred_faceMap, pred_lineSeg_map, pred_gpStroke_img = train_step(input_img, 
                                                                                             gp_token,
                                                                                             gt_bSeg_map,
                                                                                             gt_faceSeg_map,
                                                                                             input_raw,
                                                                                             group_depthNormals,
                                                                                             full_stroke_map,
                                                                                             gp_label)

        # display training loss
        if step % hyper_params['dispLossStep'] == 0:
            train_logger.info('Training loss at step {} is: {}'.format(step, train_total_loss_val))
            with train_summary_writer.as_default():
                tf.summary.scalar('train_total_loss', train_total_loss_metric.result(), step=step)
                tf.summary.scalar('train_bSegL2_loss', train_baseSeg_loss_metric.result(), step=step)
                tf.summary.scalar('train_faceSegL2_loss', train_faceSeg_loss_metric.result(), step=step)
                train_total_loss_metric.reset_states()
                train_baseSeg_loss_metric.reset_states()
                train_faceSeg_loss_metric.reset_states()
                # slice input and output images
                train_maps = tf.reshape(tf.slice(gmap_raw, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]), [-1, 256, 256, 5])
                train_gt_depth_map = tf.slice(train_maps, [0, 0, 0, 1], [-1, -1, -1, 1])
                tf.summary.image('train_gt_depth', train_gt_depth_map, step=step, max_outputs=5)
                train_gt_normal_map = tf.slice(train_maps, [0, 0, 0, 2], [-1, -1, -1, 3])
                tf.summary.image('train_gt_normal', train_gt_normal_map, step=step, max_outputs=5)
                train_gt_line_map = tf.reshape(tf.slice(gt_bSeg_map, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]), [-1, 256, 256, 1])
                tf.summary.image('train_gt_bSeg', train_gt_line_map, step=step, max_outputs=5)
                train_pred_line_map = tf.reshape(tf.slice(pred_lineSeg_map, [0, 0, 0, 0, 0], [1, -1, -1, -1, 1]),
                                                   [-1, 256, 256, 1])
                tf.summary.image('train_pred_bSeg', train_pred_line_map, step=step, max_outputs=5)
                train_gt_face_map = tf.reshape(tf.slice(gt_faceSeg_map, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]), [-1, 256, 256, 1])
                tf.summary.image('train_gt_faceSeg', train_gt_face_map, step=step, max_outputs=5)
                train_pred_face_map = tf.reshape(tf.slice(pred_faceMap, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]), [-1, 256, 256, 1])
                tf.summary.image('train_pred_faceSeg', train_pred_face_map, step=step, max_outputs=5)
                tf.summary.image('train_fullStroke', full_stroke_map, step=step, max_outputs=4)
                train_gt_gp_strokes = tf.reshape(tf.slice(gt_gp_strokes, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                 [-1, 256, 256, 1])
                tf.summary.image('train_gt_gpStrokes', train_gt_gp_strokes, step=step, max_outputs=5)
                train_pred_gp_strokes = tf.reshape(tf.slice(pred_gpStroke_img, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                   [-1, 256, 256, 1])
                tf.summary.image('train_pred_gpStrokes', train_pred_gp_strokes, step=step, max_outputs=5)
                train_gp_depth = tf.slice(group_depthNormals, [0, 0, 0, 0, 0], [1, -1, -1, -1, 1])
                train_gp_depth = tf.reshape(train_gp_depth, [-1, 256, 256, 1])
                tf.summary.image('train_gp_depth', train_gp_depth, step=step, max_outputs=5)
                train_gp_normal = tf.slice(group_depthNormals, [0, 0, 0, 0, 1], [1, -1, -1, -1, 3])
                train_gp_normal = tf.reshape(train_gp_normal, [-1, 256, 256, 3])
                tf.summary.image('train_gp_normal', train_gp_normal, step=step, max_outputs=5)

        # eval step
        if step % hyper_params['exeValStep'] == 0 and step > 0:
            val_total_loss_metric.reset_states()
            val_baseSeg_loss_metric.reset_states()
            val_faceSeg_loss_metric.reset_states()
            try:
                while True:
                    val_input_raw, val_glabel_raw, val_gmap_raw, _, _, _, _, val_global_nd, val_gbseg_raw, \
                    val_nb_gps, val_nb_strokes, val_pregp_raw, val_startidx_raw, val_gfseg_raw = readerEval.next()

                    val_input_img, val_gp_token, val_full_stroke_map, val_group_depthNormals, val_gt_gpStrokes, \
                    val_gt_bSeg_map, val_gt_fSeg_map, val_gp_label = cook_raw(modelAESSG, val_input_raw, val_glabel_raw, 
                                                                              val_gmap_raw, val_nb_gps, val_nb_strokes,
                                                                              hyper_params['d_model'], 
                                                                              val_global_nd, val_gbseg_raw, val_pregp_raw, 
                                                                              val_startidx_raw, val_gfseg_raw)

                    _, val_pred_faceMap, val_pred_lineSegMap, val_pred_gpStroke_img = eval_step(val_input_img,
                                                                                                val_gp_token,
                                                                                                val_gt_bSeg_map,
                                                                                                val_gt_fSeg_map,
                                                                                                val_input_raw,
                                                                                                val_group_depthNormals,
                                                                                                val_full_stroke_map,
                                                                                                val_gp_label)

            except StopIteration:
                train_logger.info('Validating loss at step {} is: {}'.format(step, val_total_loss_metric.result()))

                with test_summary_writer.as_default():
                    tf.summary.scalar('val_total_loss', val_total_loss_metric.result(), step=step)
                    tf.summary.scalar('val_baseSegL2_loss', val_baseSeg_loss_metric.result(), step=step)
                    tf.summary.scalar('val_faceSegL2_loss', val_faceSeg_loss_metric.result(), step=step)

                    # slice input and output images
                    eval_gt_bSeg_map = tf.reshape(tf.slice(val_gt_bSeg_map, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                  [-1, 256, 256, 1])
                    tf.summary.image('val_gt_lineSeg', eval_gt_bSeg_map, step=step, max_outputs=5)
                    eval_pred_bSeg_map = tf.reshape(tf.slice(val_pred_lineSegMap, [0, 0, 0, 0, 0], [1, -1, -1, -1, 1]),
                                                      [-1, 256, 256, 1])
                    tf.summary.image('val_pred_lineSeg', eval_pred_bSeg_map, step=step, max_outputs=5)

                    val_maps = tf.reshape(tf.slice(val_gmap_raw, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                          [-1, 256, 256, 5])
                    eval_gt_depth_map = tf.slice(val_maps, [0, 0, 0, 1], [-1, -1, -1, 1])
                    tf.summary.image('val_gt_depth', eval_gt_depth_map, step=step, max_outputs=5)
                    eval_gt_normal_map = tf.slice(val_maps, [0, 0, 0, 2], [-1, -1, -1, 3])
                    tf.summary.image('val_gt_normal', eval_gt_normal_map, step=step, max_outputs=5)
                    
                    tf.summary.image('val_fullStroke', val_full_stroke_map, step=step, max_outputs=4)
                    eval_gt_gpStroke = tf.reshape(tf.slice(val_gt_gpStrokes, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                  [-1, 256, 256, 1])
                    tf.summary.image('val_gt_gpStrokes', eval_gt_gpStroke, step=step, max_outputs=5)
                    eval_pred_gpStroke = tf.reshape(tf.slice(val_pred_gpStroke_img, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                    [-1, 256, 256, 1])
                    tf.summary.image('val_pred_gpStrokes', eval_pred_gpStroke, step=step, max_outputs=5)
                    
                    eval_gp_depth = tf.slice(val_group_depthNormals, [0, 0, 0, 0, 0], [1, -1, -1, -1, 1])
                    eval_gp_depth = tf.reshape(eval_gp_depth, [-1, 256, 256, 1])
                    tf.summary.image('val_gp_depth', eval_gp_depth, step=step, max_outputs=5)
                    eval_gp_normal = tf.slice(val_group_depthNormals, [0, 0, 0, 0, 1], [1, -1, -1, -1, 3])
                    eval_gp_normal = tf.reshape(eval_gp_normal, [-1, 256, 256, 3])
                    tf.summary.image('val_gp_normal', eval_gp_normal, step=step, max_outputs=5)

                    eval_gt_fSeg_map = tf.reshape(tf.slice(val_gt_fSeg_map, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]), 
                                                    [-1, 256, 256, 1])
                    tf.summary.image('val_gt_faceMap', eval_gt_fSeg_map, step=step, max_outputs=5)
                    eval_pred_fSeg_map = tf.reshape(tf.slice(val_pred_faceMap, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]), 
                                                      [-1, 256, 256, 1])
                    tf.summary.image('val_pred_cornerMap', eval_pred_fSeg_map, step=step, max_outputs=5)

        # save model
        if step % hyper_params['saveModelStep'] == 0 and step > 0:
            ckpt_save_path = ckpt_manager.save()
            train_logger.info('Save model at step: {:d} to file: {}'.format(step, ckpt_save_path))


def test_net():
    # set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('-----------------Begin testing: -------------------')
    
    # reader
    readerTest = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=hyper_params['batchSize'], shuffle=False,
                               raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=False, prefix='test')
    
    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG, transformer=transformer, regressor=regressor)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=50)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        test_logger.info('restore from the checkoiint {}'.format(ckpt_manager.latest_checkpoint))
        
    # save model
    tf.saved_model.save(regressor, os.path.join(output_folder, 'regNet'))
    test_logger.info('Write model to regNet')
    
    tf.saved_model.save(transformer, os.path.join(output_folder, 'gpTFNet'))
    test_logger.info('Write model to gpTFNet')

    tf.saved_model.save(modelAESSG, os.path.join(output_folder, 'embedNet'))
    test_logger.info('Write model to embedNet')
    

if __name__ == '__main__':

    # Set output folder
    timeSufix = time.strftime(r'%Y%m%d_%H%M%S')
    output_folder = hyper_params['outDir'] + '_{}'.format(timeSufix)
    if tf.io.gfile.exists(output_folder):
        tf.io.gfile.rmtree(output_folder)
    tf.io.gfile.makedirs(output_folder)

    # Set logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_folder, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Begin training
    if hyper_params['status'] == 'train':
        train_net()
    elif hyper_params['status'] == 'test':
        test_net()
