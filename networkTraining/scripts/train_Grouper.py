#
# Project Free2CAD
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2022. All Rights Reserved.
#
# ==============================================================================
"""Grouping Transformer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import logging
import argparse
import time
import datetime

import tensorflow as tf
import numpy as np
from loader import GPRegTFReader
from network import GpTransformer, AutoencoderEmbed
from tensorflow.keras.preprocessing import image

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
    'rootSize': 32,
    'status:': 'train',
    'embed_ckpt': '',
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
parser.add_argument('--embed_ckpt', required=True, help='stroke embedding checkpoint', type=str)
parser.add_argument('--devices', required=True, help='GPU device indices', type=str)
parser.add_argument('--ckpt', help='checkpoint path', type=str, default='')
parser.add_argument('--cnt', help='continue training flag', type=bool, default=False)
parser.add_argument('--status', help='training or testing flag', type=str, default='train')
parser.add_argument('--d_model', help='codeSize and bottleneck size', type=int, default=256)
parser.add_argument('--bSize', help='batch size', type=int, default=16)
parser.add_argument('--maxS', help='maximum step size', type=int, default=3)

args = parser.parse_args()
hyper_params['dbDir'] = args.dbDir
hyper_params['outDir'] = args.outDir
hyper_params['device'] = args.devices
hyper_params['ckpt'] = args.ckpt
hyper_params['cnt'] = args.cnt
hyper_params['status'] = args.status
hyper_params['embed_ckpt'] = args.embed_ckpt
hyper_params['d_model'] = args.d_model
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


def loss_fn(real, pred):
    # create mask
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, -1.0)), tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(real, pred)  # remove sigmoid from the last network layer
    loss *= mask

    nb_elem = tf.reduce_sum(mask)
    loss_val = tf.reduce_sum(loss) / nb_elem
    # loss_val = tf.reduce_mean(loss)

    pred_sigmoid_masked = tf.round(tf.math.sigmoid(pred)) * mask
    real_masked = real * mask
    acc_val = tf.reduce_sum(tf.math.abs(pred_sigmoid_masked - real_masked)) / nb_elem

    return loss_val, 1.0 - acc_val


def get_predicted_gp(inp, tar, gt_label, label_pred, allStroke_map):
    
    # mask
    mask = tf.cast(tf.math.logical_not(tf.math.equal(gt_label, -1.0)), tf.float32)

    # calculate the real nb_g, nb_s
    # inp: [N, nb_s, 256]
    inp_sum = tf.reduce_sum(inp, axis=2)
    inp_sum_mask = tf.cast(tf.logical_not(tf.math.equal(inp_sum, -2.0 * 256)), tf.int32)  # [N, nb_s]
    nb_strokes = tf.reduce_sum(inp_sum_mask, axis=1)  # [N]

    # tar: [N, nb_g, 256]
    tar_sum = tf.reduce_sum(tar, axis=2)
    tar_sum_mask = tf.cast(tf.logical_not(tf.math.equal(tar_sum, -2.0 * 256)), tf.int32)  # [N, nb_g]
    nb_groups = tf.reduce_sum(tar_sum_mask, axis=1)  # [N]

    target_gp_nb = tf.shape(tar)[1]

    # sigmoid, round
    label_pred = tf.sigmoid(label_pred) * mask  # [N, nb_gp, nb_s]
    
    # assemble strokes
    nb_batch = tf.shape(gt_label)[0]
    gp_stroke_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for itr in range(nb_batch):
        cur_nb_s = nb_strokes[itr]
        cur_nb_g = nb_groups[itr]
        cur_strokes = tf.slice(allStroke_map, [itr, 0, 0, 0], [1, -1, -1, cur_nb_s])  # [1, 256, 256, nb_s]
        cur_strokes = tf.transpose(cur_strokes, [3, 1, 2, 0])  # [nb_s, 256, 256, 1]
        cur_strokes = tf.subtract(1.0, cur_strokes)  # inverse stroke value: background-0.0, strokes-1.0
        cur_gp_labels = tf.slice(label_pred, [itr, 0, 0], [1, cur_nb_g, cur_nb_s])  # [1, nb_g, nb_s]
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

    pred_gp_stroke = gp_stroke_list.stack()  # [N, nb_g, 256, 256, 1]

    return pred_gp_stroke


def stroke_acc(label_pred, gt_label):
    gt_label = gt_label[:, :-1, :]
    gt_label_idx = tf.argmax(gt_label, 1)
    
    label_pred = tf.sigmoid(label_pred)
    label_pred = label_pred[:, :-1, :]
    label_pred_idx = tf.argmax(label_pred, 1)

    acc_val = tf.reduce_mean(tf.where(tf.equal(gt_label_idx, label_pred_idx), 1.0, 0.0))

    return acc_val


# define model
modelAESSG = AutoencoderEmbed(code_size=hyper_params['d_model'], x_dim=hyper_params['imgSize'],
                              y_dim=hyper_params['imgSize'], root_feature=32)
embS_ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG)
embS_ckpt_manager = tf.train.CheckpointManager(embS_ckpt, hyper_params['embed_ckpt'], max_to_keep=30)
if embS_ckpt_manager.latest_checkpoint:
    embS_ckpt.restore(embS_ckpt_manager.latest_checkpoint)
    print('restore stroke network from the checkpoint {}'.format(embS_ckpt_manager.latest_checkpoint))

transformer = GpTransformer(num_layers=hyper_params['nb_layers'], d_model=hyper_params['d_model'],
                            num_heads=hyper_params['nb_heads'], dff=hyper_params['d_ff'],
                            pe_input=hyper_params['nb_stroke_max'], pe_target=hyper_params['nb_gp_max'],
                            rate=hyper_params['drop_rate'])

# define reader
readerTrain = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=hyper_params['batchSize'], shuffle=True,
                          raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=True, prefix='train')
readerEval = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=hyper_params['batchSize'], shuffle=False,
                         raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=False, prefix='eval')

# optimizer
# learning_rate = CustomSchedule(hyper_params['d_model'])
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

val_loss_metric = tf.keras.metrics.Mean(name='Validate_loss', dtype=tf.float32)
val_acc_metric = tf.keras.metrics.Mean(name='Validate_acc', dtype=tf.float32)
val_sacc_metric = tf.keras.metrics.Mean(name='Validate_sacc', dtype=tf.float32)
train_loss_metric = tf.keras.metrics.Mean(name='Train_loss', dtype=tf.float32)
train_acc_metric = tf.keras.metrics.Mean(name='Train_acc', dtype=tf.float32)
train_sacc_metric = tf.keras.metrics.Mean(name='Train_sacc', dtype=tf.float32)

train_step_signature = [
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256), dtype=tf.float32),  # input
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256), dtype=tf.float32),  # grouping token
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, None), dtype=tf.float32),  # grouping label
    tf.TensorSpec(shape=(hyper_params['batchSize'], 256, 256, None), dtype=tf.float32),  # all strokes
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, label, allStroke_map):
    tar_inp = tar
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss_val, acc_val = loss_fn(label, predictions)
        pred_gp_strokes = get_predicted_gp(inp, tar_inp, label, predictions, allStroke_map)
        sacc_val = stroke_acc(predictions, label)

    gradients = tape.gradient(loss_val, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss_metric.update_state(loss_val)
    train_acc_metric.update_state(acc_val)
    train_sacc_metric.update_state(sacc_val)

    return loss_val, acc_val, pred_gp_strokes, sacc_val


eval_step_signature = [
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256), dtype=tf.float32),  # input
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, 256), dtype=tf.float32),  # grouping token
    tf.TensorSpec(shape=(hyper_params['batchSize'], None, None), dtype=tf.float32),  # grouping label
    tf.TensorSpec(shape=(hyper_params['batchSize'], 256, 256, None), dtype=tf.float32),  # all strokes
]


@tf.function(input_signature=eval_step_signature)
def eval_step(inp, tar, label, allStroke_map):
    tar_inp = tar
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = transformer(inp, tar_inp,
                                 False,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    loss_val, acc_val = loss_fn(label, predictions)
    val_pred_gp_stroke = get_predicted_gp(inp, tar_inp, label, predictions, allStroke_map)
    val_sacc_val = stroke_acc(predictions, label)

    val_loss_metric.update_state(loss_val)
    val_acc_metric.update_state(acc_val)
    val_sacc_metric.update_state(val_sacc_val)

    sacc_val = stroke_acc(predictions, label)

    return loss_val, acc_val, tf.sigmoid(predictions), val_pred_gp_stroke, sacc_val


def assemble_gp(pred_label, full_stroke_input):
    # pred_label [1, 1, nb_stroke]
    # full_stroke_input [1, 256, 256, nb_stroke]

    nb_stroke = tf.shape(pred_label)[2]

    cur_strokes = tf.transpose(full_stroke_input, [3, 1, 2, 0])  # [nb_s, 256, 256, 1]
    cur_gp_labels = tf.reshape(pred_label, [1, nb_stroke])  # [1, nb_stroke]
    cur_stroke_rep = tf.reshape(cur_strokes, [1, nb_stroke, 256, 256, 1])  # [1, nb_stroke, 256, 256, 1]
    cur_gp_label_rep = tf.repeat(tf.expand_dims(cur_gp_labels, axis=2), 256, axis=2)  # [1, nb_stroke, 256]
    cur_gp_label_rep = tf.repeat(tf.expand_dims(cur_gp_label_rep, axis=3), 256, axis=3)  # [1, nb_stroke, 256, 256]
    cur_gp_label_rep = tf.reshape(cur_gp_label_rep, [1, nb_stroke, 256, 256, 1])  # [1, nb_stroke, 256, 256, 1]

    gp_strokes_sel = cur_gp_label_rep * cur_stroke_rep  # [1, nb_s, 256, 256, 1]
    gp_stroke_sum = tf.reduce_sum(gp_strokes_sel, axis=1)  # [1, 256, 256, 1]
    label_rep_sum = tf.reduce_sum(cur_gp_label_rep, axis=1)  # [1, 256, 256, 1]
    gp_strokes = tf.where(tf.math.equal(gp_stroke_sum, label_rep_sum), 1.0, 0.0)  # [1, 256, 256, 1]

    return gp_strokes


def test_iterative_decode_step(inp, label, test_true_acc, full_stroke_maps):
    # inp:      [1, nb_s, 256]
    # label:    [1, nb_g, nb_s]
    # fullsmap: [1, 256, 256, nb_s]
    
    # assemble start group token
    nb_stroke = tf.shape(inp)[1]
    stroke_idx = np.arange(0, nb_stroke)
    cur_inp = inp
    cur_nb_s = nb_stroke
    cur_stroke_idx = np.arange(0, cur_nb_s)
    full_prediction = []
    gp_token = tf.fill([1, 1, 256], -1.0)
    while cur_nb_s > 1:
        
        # from START token, we call Transformer for K steps, expect K<=3
        if full_prediction:
           current_predicted_stroke_labels = tf.concat(full_prediction, axis=1)  # [1, nb_g', nb_stroke]
           current_predicted_stroke_labels_combined = tf.reduce_max(current_predicted_stroke_labels, axis=1) # [1, nb_stroke]
           current_predicted_stroke_labels_combined_np = current_predicted_stroke_labels_combined.numpy()
           current_idx_del = np.where(current_predicted_stroke_labels_combined_np==0)[1]  # where the strokes are not selected
           full_stroke_maps_np = full_stroke_maps.numpy()
           full_stroke_map_grouped = np.delete(full_stroke_maps_np, current_idx_del, axis=3)  # [1, 256, 256, ng']
           full_stroke_map_grouped_img = tf.reshape(tf.reduce_min(tf.convert_to_tensor(full_stroke_map_grouped, dtype=tf.float32), axis=3), 
                                                    [1, 256, 256, 1])  # [1, 256, 256, 1]
           gp_token = modelAESSG.encoder(full_stroke_map_grouped_img, training=False)  # [1, 256]
           gp_token = tf.reshape(gp_token, [1, 1, 256])

        predicts = []
        nb_max_try = 6
        for itr in range(nb_max_try):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(cur_inp, gp_token)
            predictions, _ = transformer(cur_inp, gp_token, False, enc_padding_mask,
                                        combined_mask, dec_padding_mask)  # [batch, nb_group, nb_stroke]

            # check the last predictions
            pred_sigmoid_round = tf.round(tf.math.sigmoid(predictions))
            pred_last_labels = pred_sigmoid_round[:, -1:, :]  # [batch, 1, nb_stroke]

            # reach the end
            if tf.reduce_sum(pred_last_labels) < 1.0:
                break

            # add to the global prediction
            cur_global_label = np.zeros(nb_stroke)
            pred_last_labels_np = pred_last_labels.numpy()
            cur_global_label[stroke_idx[cur_stroke_idx[np.where(pred_last_labels_np==1)[2]]]] = 1.0
            cur_global_label_tf = tf.reshape(tf.convert_to_tensor(cur_global_label, dtype=tf.float32), [1, 1, -1])  # [1, 1, nb_stroke]
            full_prediction.append(cur_global_label_tf)

            # calculate the new gp token and add to gp_token
            last_gp_strokes_sel = assemble_gp(cur_global_label_tf, full_stroke_maps)  # [1, 256, 256, 1]
            last_gp_strokes = tf.where(last_gp_strokes_sel>0.5, 1.0, 0.0)  # [1, 256, 256, 1]
            last_gp_embed = modelAESSG.encoder(last_gp_strokes, training=False)  # [1, 256]
            last_gp_embed = tf.reshape(last_gp_embed, [1, 1, 256])  # [1, 1, 256]
            
            gp_token = tf.concat([gp_token, last_gp_embed], axis=1)
            predicts.append(pred_last_labels)

        # check empty of predicts
        if not predicts:
            break

        # update the stroke set and the index
        itr_pred_labels = tf.concat(predicts, axis=1)  # [1, nb_group, cur_nb_stroke]
        itr_pred_labels_combined = tf.reduce_max(itr_pred_labels, axis=1)  # [1, cur_nb_stroke]
        itr_pred_labels_combined_np = itr_pred_labels_combined.numpy() # [1, cur_nb_stroke]
        idx_del = np.where(itr_pred_labels_combined_np==1)[1]
        idx_keep = np.where(itr_pred_labels_combined_np==0)[1]

        cur_inp_np = cur_inp.numpy()  # [1, cur_nb_s, 256]
        cur_inp_np = np.delete(cur_inp_np, idx_del, axis=1)  # [1, new_cur_nb_s, 256]
        cur_inp = tf.convert_to_tensor(cur_inp_np, dtype=tf.float32)  # [1, new_cur_nb_s, 256]
        cur_nb_s = np.shape(idx_keep)[0]
        cur_stroke_idx = cur_stroke_idx[idx_keep]

    # calculate accuracy
    # concat to get the predicted labels
    if not full_prediction:
        full_prediction.append(tf.fill([1, 1, nb_stroke], 0.0))
    true_pred_labels = tf.concat(full_prediction, axis=1)  # [batch, nb_group, nb_stroke]

    # remove the padded 0 at the end of label
    label = label[:, :-1, :]
    # find the maximum value and pad to measure the accuracy
    tar_nb_gp = tf.maximum(label.shape[1], true_pred_labels.shape[1])

    # pad label
    if label.shape[1] < tar_nb_gp:
        label = tf.pad(label, [[0, 0], [0, tar_nb_gp - label.shape[1]], [0, 0]], constant_values=0.0)

    if true_pred_labels.shape[1] < tar_nb_gp:
        true_pred_labels = tf.pad(true_pred_labels,
                                  [[0, 0], [0, tar_nb_gp - true_pred_labels.shape[1]], [0, 0]],
                                  constant_values=0.0)

    true_acc_val = 1.0 - tf.reduce_mean(tf.math.abs(true_pred_labels - label))
    test_true_acc.update_state(true_acc_val)

    return true_acc_val, true_pred_labels


def test_golden_decode_step(inp, tar, label, test_true_acc, full_stroke_maps):
    # inp:      [1, nb_s, 256]
    # label:    [1, nb_g+1, nb_s], all zero at the end
    # tar:      [1, nb_g+1, 256], all -1 at the begining
    # fullsmap: [1, 256, 256, nb_s]

    label = label[:, :-1, :]  # [1, nb_g, nb_stroke]
    tar = tar[:, 1:, :]  # [1, nb_g, 256]
    nb_gp = tf.shape(tar)[1]
    k = hyper_params['maxS']
    nb_stroke = tf.shape(inp)[1]
    stroke_idx = np.arange(0, nb_stroke)
    cur_inp = inp
    cur_nb_g = nb_gp
    cur_start_gid = 0
    cur_stroke_idx = np.arange(0, nb_stroke)
    start_token = tf.fill([1, 1, 256], -1.0)
    full_prediction = []
    itr_cnt = 0
    while(cur_nb_g > 0):
        
        if itr_cnt > 0:
            current_predicted_stroke_labels = label[:, 0:itr_cnt*k, :]  # [1, ng', nb_stroke]
            current_predicted_stroke_labels_combined = tf.reduce_max(current_predicted_stroke_labels, axis=1) # [1, nb_stroke]
            current_predicted_stroke_labels_combined_np = current_predicted_stroke_labels_combined.numpy()
            current_idx_del = np.where(current_predicted_stroke_labels_combined_np==0)[1]  # where the strokes are not selected
            full_stroke_maps_np = full_stroke_maps.numpy()
            full_stroke_map_grouped = np.delete(full_stroke_maps_np, current_idx_del, axis=3)  # [1, 256, 256, ng']
            full_stroke_map_grouped_img = tf.reshape(tf.reduce_min(tf.convert_to_tensor(full_stroke_map_grouped, dtype=tf.float32), axis=3), 
                                                     [1, 256, 256, 1])  # [1, 256, 256, 1]
            start_token = modelAESSG.encoder(full_stroke_map_grouped_img, training=False)  # [1, 256]
            start_token = tf.reshape(start_token, [1, 1, 256])

        # from start token, we call Transformer for nb_g / K times
        cur_gp_token = tar[:, cur_start_gid:cur_start_gid+k, :]  # [1, k, 256]
        cur_gp_token = tf.concat([start_token, cur_gp_token], axis=1)  # [1, k+1, 256]
        
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(cur_inp, cur_gp_token)
        predictions, _ = transformer(cur_inp, cur_gp_token, False, enc_padding_mask,
                                     combined_mask, dec_padding_mask)  # [1, k+1, nb_stroke]
        
        pred_sigmoid_round = tf.round(tf.math.sigmoid(predictions))   # [1, k+1, nb_stroke]
        cur_nb_pred = tf.shape(predictions)[1]  # k+1
        for itr in range(0, cur_nb_pred-1):
            pred_last_labels = pred_sigmoid_round[:, itr:itr+1, :]  # [1, 1, nb_stroke]
            # add to the global prediction
            cur_global_label = np.zeros(nb_stroke)
            pred_last_labels_np = pred_last_labels.numpy()
            cur_global_label[stroke_idx[cur_stroke_idx[np.where(pred_last_labels_np==1)[2]]]] = 1.0
            cur_global_label_tf = tf.reshape(tf.convert_to_tensor(cur_global_label, dtype=tf.float32), [1, 1, -1])  # [1, 1, nb_stroke]
            full_prediction.append(cur_global_label_tf)

        # update the stroke set and the index
        cur_gt_label = label[:, :cur_start_gid+k, :]  # [1, k, cur_nb_stroke]
        cur_gt_label_combined = tf.reduce_max(cur_gt_label, axis=1)  # [1, cur_nb_stroke]
        cur_gt_label_combined_np = cur_gt_label_combined.numpy()  # [1, cur_nb_stroke]
        idx_del = np.where(cur_gt_label_combined_np==1)[1]
        idx_keep = np.where(cur_gt_label_combined_np==0)[1]
        
        cur_inp_np = inp.numpy()  # [1, cur_nb_s, 256]
        cur_inp_np = np.delete(cur_inp_np, idx_del, axis=1)  # [1, new_cur_nb_s, 256]
        cur_inp = tf.convert_to_tensor(cur_inp_np, dtype=tf.float32)  # [1, new_cur_nb_s, 256]
        cur_stroke_idx = stroke_idx[idx_keep]

        cur_nb_g = cur_nb_g - k
        cur_start_gid = cur_start_gid + k

        itr_cnt += 1

    # calculate accuracy

    if not full_prediction:
        full_prediction.append(tf.fill([1, 1, nb_stroke], 0.0))
    true_pred_labels = tf.concat(full_prediction, axis=1)  # [batch, nb_group, nb_stroke]

    # find the maximum value and pad to measure the accuracy
    tar_nb_gp = tf.maximum(label.shape[1], true_pred_labels.shape[1])

    # pad label
    if label.shape[1] < tar_nb_gp:
        label = tf.pad(label, [[0, 0], [0, tar_nb_gp - label.shape[1]], [0, 0]], constant_values=0.0)

    if true_pred_labels.shape[1] < tar_nb_gp:
        true_pred_labels = tf.pad(true_pred_labels,
                                  [[0, 0], [0, tar_nb_gp - true_pred_labels.shape[1]], [0, 0]],
                                  constant_values=0.0)

    true_acc_val = 1.0 - tf.reduce_mean(tf.math.abs(true_pred_labels - label))
    test_true_acc.update_state(true_acc_val)

    return true_acc_val, true_pred_labels


# with k=maxS
def cook_raw(net, input_raw, label_raw, pregp_raw, startidx_raw, nb_gps, nb_strokes, gp_embSize, gmap_raw, global_nd):
    gp_start_token = tf.fill([1, gp_embSize], -1.0)
    nb_batch = tf.shape(input_raw)[0]

    input_cook_list = []
    gp_token_cook_list = []
    label_cook_list = []
    gp_stroke_list = []
    recons_stroke_cook_list = []
    gp_recons_stroke_list = []
    gp_context_list = []
    gp_context_recons_list = []
    full_stroke_list = []
    gp_depthNormal_cook_list = []
    for itr in range(nb_batch):
        # get group and stroke numbers
        nb_gp = min(nb_gps[itr], hyper_params['maxS'])
        nb_stroke = nb_strokes[itr]
        start_idx = tf.reshape(tf.slice(startidx_raw, [itr, 0], [1, 1]), [])  # []

        # get slice data
        cur_input = tf.slice(input_raw, [itr, 0, 0, 0], [1, -1, -1, nb_stroke])
        cur_label = tf.slice(label_raw, [itr, 0, 0], [1, nb_gp, nb_stroke])
        cur_label = tf.reshape(cur_label, [nb_gp, nb_stroke])   # [nb_gp, nb_stroke]
        cur_pregp = tf.slice(pregp_raw, [itr, 0, 0], [1, -1, -1]) # [1, 256, 256]
        cur_gp_maps = tf.slice(gmap_raw, [itr, 0, 0, 0, 0], [1, nb_gp, -1, -1, -1])  # [1, nb_gp, 256, 256, 5]
        cur_gp_maps = tf.reshape(cur_gp_maps, [nb_gp, 256, 256, 5])  # [nb_g, 256, 256, 5]
        cur_gp_depthNormal_maps = tf.slice(cur_gp_maps, [0, 0, 0, 1], [-1, -1, -1, 4])  # [nb_g, 256, 256, 4]
        cur_global_gp_depthNormal_maps = tf.slice(global_nd, [itr, 0, 0, 0], [1, -1, -1, -1])  # [1, 256, 256, 4]


        # full stroke input
        cur_full_stroke = tf.reshape(tf.reduce_min(cur_input, axis=3), [1, 256, 256, 1])  # [1, 256, 256, 1]
        full_stroke_list.append(cur_full_stroke)

        # stroke embedding
        input_trans = tf.transpose(cur_input, [3, 1, 2, 0])    # [nb_stroke, 256, 256, 1]
        input_cook = net.encoder(input_trans, training=False)  # [nb_stroke, 256]
        recons_input_cook = net.decoder(input_cook)   # [nb_stroke, 256, 256, 1]

        # new scheme: select all the strokes within one group and forward network
        # [nb_g, nb_s, 256, 256, 1]
        input_trans_rep = tf.repeat(tf.expand_dims(input_trans, axis=0), nb_gp, axis=0)
        # [nb_g, nb_s, 256, 256, 1]
        label_rep = tf.repeat(tf.expand_dims(cur_label, axis=2), gp_embSize, axis=2)
        label_rep = tf.repeat(tf.expand_dims(label_rep, axis=3), gp_embSize, axis=3)
        label_rep = tf.reshape(label_rep, [nb_gp, nb_stroke, gp_embSize, gp_embSize, 1])
        gp_strokes_sel = label_rep * input_trans_rep  # [nb_g, nb_s, 256, 256, 1]
        gp_stroke_sum = tf.reduce_sum(gp_strokes_sel, axis=1)  # [nb_g, 256, 256, 1]
        label_rep_sum = tf.reduce_sum(label_rep, axis=1)   # [nb_g, 256, 256, 1]
        gp_strokes = tf.where(tf.math.equal(gp_stroke_sum, label_rep_sum), 1.0, 0.0)  # [nb_g, 256, 256, 1]
        gp_embed = net.encoder(gp_strokes, training=False)  # [nb_g, 256]
        gp_recons_strokes = net.decoder(gp_embed, training=False)  # [nb_g, 256, 256, 1]

        # add start and end token
        cur_pregp = tf.reshape(cur_pregp, [1, gp_embSize, gp_embSize, 1])  # [1, 256, 256, 1]
        pre_embed = net.encoder(cur_pregp, training=False)  # [1, 256]
        cur_pregp_recons = net.decoder(pre_embed, training=False)  # [1, 256, 256, 1]

        if start_idx > 0:
            gp_start_token = pre_embed
        gp_cook = tf.concat([gp_start_token, gp_embed], axis=0)

        # label: add end group label (all zeros)
        label_cook = tf.concat([cur_label, tf.fill([1, nb_stroke], 0.0)], axis=0)

        # depth, normal maps
        cur_gp_depthNormal_cook = tf.concat([cur_global_gp_depthNormal_maps,
                                             cur_gp_depthNormal_maps], axis=0)  # [nb_gp+1, 256, 256, 4]

        # padding
        target_stroke_nb = tf.shape(input_raw)[3]
        # target_gp_nb = tf.shape(label_raw)[1]
        target_gp_nb = hyper_params['maxS']
        assert (target_stroke_nb == tf.shape(label_raw)[2])

        input_cook_shape = tf.shape(input_cook)
        input_cook_pad = tf.pad(input_cook, [[0, target_stroke_nb - input_cook_shape[0]], [0, 0]], constant_values=-2.0)
        input_cook_pad = tf.reshape(input_cook_pad, [1, -1, input_cook_shape[1]])
        input_cook_list.append(input_cook_pad)

        # reconstructed strokes, depth and normal maps
        recons_input_cook_shape = tf.shape(recons_input_cook)  # [nb_stroke, 256, 256, 1]
        recons_input_cook_pad = tf.pad(recons_input_cook,
                                       [[0, target_stroke_nb - recons_input_cook_shape[0]], [0, 0], [0, 0], [0, 0]],
                                       constant_values=-2.0)
        recons_input_cook_pad = tf.reshape(recons_input_cook_pad, [1, -1, 256, 256, 1])
        recons_stroke_cook_list.append(recons_input_cook_pad)

        gp_cook_shape = tf.shape(gp_cook)
        gp_cook_pad = tf.pad(gp_cook, [[0, target_gp_nb - gp_cook_shape[0] + 1], [0, 0]], constant_values=-2.0)
        gp_cook_pad = tf.reshape(gp_cook_pad, [1, -1, gp_cook_shape[1]])
        gp_token_cook_list.append(gp_cook_pad)

        label_cook_shape = tf.shape(label_cook)
        label_cook_pad = tf.pad(label_cook,
                                [[0, target_gp_nb - label_cook_shape[0] + 1],
                                 [0, target_stroke_nb - label_cook_shape[1]]], constant_values=-1.0)
        label_cook_pad = tf.reshape(label_cook_pad, [1, -1, target_stroke_nb])
        label_cook_list.append(label_cook_pad)

        # [nb_gp, 256, 256, 1]
        gp_stroke_pad = tf.pad(gp_strokes, [[0, target_gp_nb - nb_gp + 1], 
                                            [0, 0], [0, 0], [0, 0]], constant_values=0.0)
        gp_stroke_pad = tf.reshape(gp_stroke_pad, [1, -1, 256, 256, 1])
        gp_stroke_list.append(gp_stroke_pad)
        
        gp_recons_strokes_shape = tf.shape(gp_recons_strokes)  # [nb_g, 256, 256, 1]
        gp_recons_strokes_pad = tf.pad(gp_recons_strokes, [[0, target_gp_nb - gp_recons_strokes_shape[0] + 1],
                                                           [0, 0], [0, 0], [0, 0]], constant_values=-1.0)
        gp_recons_strokes_pad = tf.reshape(gp_recons_strokes_pad, [1, -1, 256, 256, 1])
        gp_recons_stroke_list.append(gp_recons_strokes_pad)

        gp_context_list.append(cur_pregp)
        gp_context_recons_list.append(cur_pregp_recons)

        # depth, normal maps
        cur_gp_depthNormal_cook_shape = tf.shape(cur_gp_depthNormal_cook)  # [nb_gp+1, 256, 256, 4]
        cur_gp_depthNormal_cool_pad = tf.pad(cur_gp_depthNormal_cook,
                                             [[0, target_gp_nb - cur_gp_depthNormal_cook_shape[0] + 1],
                                              [0, 0], [0, 0], [0, 0]], constant_values=-1.0)  # [4, 256, 256, 4]
        cur_gp_depthNormal_cool_pad = tf.reshape(cur_gp_depthNormal_cool_pad, [1, -1, 256, 256, 4])  # [1, 4, 256, 256, 4]
        gp_depthNormal_cook_list.append(cur_gp_depthNormal_cool_pad)


    input_img = tf.concat(input_cook_list, axis=0)
    gp_label = tf.concat(label_cook_list, axis=0)
    gp_token = tf.concat(gp_token_cook_list, axis=0)
    gp_stroke_imgs = tf.concat(gp_stroke_list, axis=0)
    recons_strokes = tf.concat(recons_stroke_cook_list, axis=0)
    gp_recons_stroke_imgs = tf.concat(gp_recons_stroke_list, axis=0)
    gp_pre_context = tf.concat(gp_context_list, axis=0)
    gp_pre_context_recons = tf.concat(gp_context_recons_list, axis=0)
    full_stroke_map = tf.concat(full_stroke_list, axis=0)
    gp_depthNormals = tf.concat(gp_depthNormal_cook_list, axis=0)

    return input_img, gp_label, gp_token, gp_stroke_imgs, recons_strokes, gp_recons_stroke_imgs, gp_pre_context, gp_pre_context_recons, full_stroke_map, gp_depthNormals


# without k=maxS, true ground truth labels
def cook_raw2(net, input_raw, label_raw, nb_gps, nb_strokes, gp_embSize):
    gp_start_token = tf.fill([1, gp_embSize], -1.0)
    nb_batch = tf.shape(input_raw)[0]

    input_cook_list = []
    gp_token_cook_list = []
    label_cook_list = []
    gp_stroke_list = []
    for itr in range(nb_batch):
        # get group and stroke numbers
        nb_gp = nb_gps[itr]
        nb_stroke = nb_strokes[itr]

        # get slice data
        cur_input = tf.slice(input_raw, [itr, 0, 0, 0], [1, -1, -1, nb_stroke])
        cur_label = tf.slice(label_raw, [itr, 0, 0], [1, nb_gp, nb_stroke])
        cur_label = tf.reshape(cur_label, [nb_gp, nb_stroke])   # [nb_gp, nb_stroke]

        # stroke embedding
        input_trans = tf.transpose(cur_input, [3, 1, 2, 0])    # [nb_stroke, 256, 256, 1]
        input_cook = net.encoder(input_trans, training=False)  # [nb_stroke, 256]

        # group token
        # [nb_g, nb_s, 256, 256, 1]
        input_trans_rep = tf.repeat(tf.expand_dims(input_trans, axis=0), nb_gp, axis=0)
        # [nb_g, nb_s, 256, 256, 1]
        label_rep = tf.repeat(tf.expand_dims(cur_label, axis=2), gp_embSize, axis=2)
        label_rep = tf.repeat(tf.expand_dims(label_rep, axis=3), gp_embSize, axis=3)
        label_rep = tf.reshape(label_rep, [nb_gp, nb_stroke, gp_embSize, gp_embSize, 1])
        gp_strokes_sel = label_rep * input_trans_rep  # [nb_g, nb_s, 256, 256, 1]
        gp_stroke_sum = tf.reduce_sum(gp_strokes_sel, axis=1)  # [nb_g, 256, 256, 1]
        label_rep_sum = tf.reduce_sum(label_rep, axis=1)   # [nb_g, 256, 256, 1]
        gp_strokes = tf.where(tf.math.equal(gp_stroke_sum, label_rep_sum), 1.0, 0.0)  # [nb_g, 256, 256, 1]
        gp_embed = net.encoder(gp_strokes, training=False)  # [nb_g, 256]

        # add start and end token
        gp_cook = tf.concat([gp_start_token, gp_embed], axis=0)

        # label: add end group label (all zeros)
        label_cook = tf.concat([cur_label, tf.fill([1, nb_stroke], 0.0)], axis=0)

        # padding
        target_stroke_nb = tf.shape(input_raw)[3]
        target_gp_nb = tf.shape(label_raw)[1]
        assert (target_stroke_nb == tf.shape(label_raw)[2])

        input_cook_shape = tf.shape(input_cook)
        input_cook_pad = tf.pad(input_cook, [[0, target_stroke_nb - input_cook_shape[0]], [0, 0]], constant_values=-2.0)
        input_cook_pad = tf.reshape(input_cook_pad, [1, -1, input_cook_shape[1]])

        gp_cook_shape = tf.shape(gp_cook)
        gp_cook_pad = tf.pad(gp_cook, [[0, target_gp_nb - gp_cook_shape[0] + 1], [0, 0]], constant_values=-2.0)
        gp_cook_pad = tf.reshape(gp_cook_pad, [1, -1, gp_cook_shape[1]])

        label_cook_shape = tf.shape(label_cook)
        label_cook_pad = tf.pad(label_cook,
                                [[0, target_gp_nb - label_cook_shape[0] + 1],
                                 [0, target_stroke_nb - label_cook_shape[1]]], constant_values=-1.0)
        label_cook_pad = tf.reshape(label_cook_pad, [1, -1, target_stroke_nb])

        # [nb_gp, 256, 256, 1]
        gp_stroke_pad = tf.pad(gp_strokes, [[0, target_gp_nb - nb_gp + 1], 
                                            [0, 0], [0, 0], [0, 0]], constant_values=0.0)
        gp_stroke_pad = tf.reshape(gp_stroke_pad, [1, -1, 256, 256, 1])
        
        gp_stroke_list.append(gp_stroke_pad)
        input_cook_list.append(input_cook_pad)
        gp_token_cook_list.append(gp_cook_pad)
        label_cook_list.append(label_cook_pad)

    gp_img = tf.concat(input_cook_list, axis=0)
    gp_label = tf.concat(label_cook_list, axis=0)
    gp_token = tf.concat(gp_token_cook_list, axis=0)
    gp_stroke_imgs = tf.concat(gp_stroke_list, axis=0)

    return gp_img, gp_label, gp_token, gp_stroke_imgs


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
    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG, transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=100)

    if hyper_params['cnt']:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            train_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    # Training process
    for step in range(hyper_params['maxIter']):

        # train step
        input_raw, label_raw, gmap_raw, _, _, _, _, global_nd, _, nb_gps, nb_strokes, \
        pregp_raw, startidx_raw, _ = readerTrain.next()

        input_img, gp_label, gp_token, gp_strokes, recons_input_img, \
        recons_gp_img, pre_context_img, pre_context_recons_img, \
        full_strokeMap, gp_dnmaps = cook_raw(modelAESSG, input_raw, label_raw, pregp_raw, 
                                             startidx_raw, nb_gps, nb_strokes,
                                             hyper_params['d_model'], gmap_raw, global_nd)

        train_loss_val, train_acc_val, pred_gp_strokes, train_sacc_val = train_step(input_img, gp_token, gp_label, input_raw)

        # display training loss
        if step % hyper_params['dispLossStep'] == 0:
            train_logger.info('Training loss at step {} is: {}, acc is: {}, sacc is: {}'.
                              format(step, train_loss_val, train_acc_val, train_sacc_val))
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss_metric.result(), step=step)
                tf.summary.scalar('train_acc', train_acc_metric.result(), step=step)
                tf.summary.scalar('train_sacc', train_sacc_metric.result(), step=step)
                train_loss_metric.reset_states()
                train_acc_metric.reset_states()
                train_sacc_metric.reset_states()
                train_gt_strokes = tf.slice(input_raw, [0, 0, 0, 0], [1, -1, -1, -1])
                train_gt_strokes = tf.transpose(train_gt_strokes, [3, 1, 2, 0])
                tf.summary.image('train_gt_stroke', train_gt_strokes, step=step, max_outputs=50)
                train_recons_stroke = tf.reshape(tf.slice(recons_input_img, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]), [-1, 256, 256, 1])
                tf.summary.image('train_recons_stroke', train_recons_stroke, step=step, max_outputs=50)
                train_gt_gp_strokes = tf.reshape(tf.slice(gp_strokes, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                 [-1, 256, 256, 1])
                tf.summary.image('train_gt_gpStrokes', train_gt_gp_strokes, step=step, max_outputs=5)
                train_pred_gp_strokes = tf.reshape(tf.slice(pred_gp_strokes, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                   [-1, 256, 256, 1])
                tf.summary.image('train_pred_gpStrokes', train_pred_gp_strokes, step=step, max_outputs=5)
                train_recons_gp_strokes = tf.reshape(tf.slice(recons_gp_img, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                     [-1, 256, 256, 1])
                tf.summary.image('train_recons_gpStrokes', train_recons_gp_strokes, step=step, max_outputs=5)
                train_precontext_strokes = tf.slice(pre_context_img, [0, 0, 0, 0], [1, -1, -1, -1])  # [1, 256, 256, 1]
                tf.summary.image('train_precontext_strokes', train_precontext_strokes, step=step, max_outputs=1)
                train_precontext_recons_strokes = tf.slice(pre_context_recons_img, [0, 0, 0, 0], [1, -1, -1, -1])
                tf.summary.image('train_precontext_recons_strokes', train_precontext_recons_strokes, step=step, max_outputs=1)
                tf.summary.image('train_fullStroke', full_strokeMap, step=step, max_outputs=4)
                train_gp_depth = tf.slice(gp_dnmaps, [0, 0, 0, 0, 0], [1, -1, -1, -1, 1])
                train_gp_depth = tf.reshape(train_gp_depth, [-1, 256, 256, 1])
                tf.summary.image('train_gp_depth', train_gp_depth, step=step, max_outputs=5)
                train_gp_normal = tf.slice(gp_dnmaps, [0, 0, 0, 0, 1], [1, -1, -1, -1, 3])
                train_gp_normal = tf.reshape(train_gp_normal, [-1, 256, 256, 3])
                tf.summary.image('train_gp_normal', train_gp_normal, step=step, max_outputs=5)

        # eval step
        if step % hyper_params['exeValStep'] == 0:
            val_loss_metric.reset_states()
            val_acc_metric.reset_states()
            try:
                while True:
                    input_raw, label_raw, gpmap_raw, _, _, _, _, global_nd, _, nb_gps, nb_strokes, \
                    pregp_raw, startidx_raw, _ = readerEval.next()

                    val_gp_img, val_label, val_gp_token, val_gp_strokes, \
                    val_recons_input_img, val_recons_gp_img, val_pre_context_img, \
                    val_pre_context_recons_img, val_full_strokeMap, val_gp_dnmaps = cook_raw(modelAESSG, input_raw, 
                                                                                             label_raw, pregp_raw, 
                                                                                             startidx_raw, nb_gps, nb_strokes,
                                                                                             hyper_params['d_model'], 
                                                                                             gpmap_raw, global_nd)

                    _, _, _, val_pred_gp_strokes, _ = eval_step(val_gp_img, val_gp_token, val_label, input_raw)

            except StopIteration:
                train_logger.info('Validating loss at step {} is: {}, acc is: {}, sacc is: {}'.
                                  format(step, val_loss_metric.result(), val_acc_metric.result(), val_sacc_metric.result()))
                with test_summary_writer.as_default():
                    tf.summary.scalar('val_loss', val_loss_metric.result(), step=step)
                    tf.summary.scalar('val_acc', val_acc_metric.result(), step=step)
                    tf.summary.scalar('val_sacc', val_sacc_metric.result(), step=step)

                    eval_gt_strokes = tf.slice(input_raw, [0, 0, 0, 0], [1, -1, -1, -1])
                    eval_gt_strokes = tf.transpose(eval_gt_strokes, [3, 1, 2, 0])
                    tf.summary.image('val_gt_stroke', eval_gt_strokes, step=step, max_outputs=50)
                    eval_recons_stroke = tf.reshape(tf.slice(val_recons_input_img, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]), [-1, 256, 256, 1])
                    tf.summary.image('val_recons_stroke', eval_recons_stroke, step=step, max_outputs=50)
                    eval_gt_gpStroke = tf.reshape(tf.slice(val_gp_strokes, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                  [-1, 256, 256, 1])
                    tf.summary.image('val_gt_gpStrokes', eval_gt_gpStroke, step=step, max_outputs=5)
                    eval_pred_gpStroke = tf.reshape(tf.slice(val_pred_gp_strokes, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                    [-1, 256, 256, 1])
                    tf.summary.image('val_pred_gpStrokes', eval_pred_gpStroke, step=step, max_outputs=5)
                    eval_recons_gp_strokes = tf.reshape(tf.slice(val_recons_gp_img, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1]),
                                                     [-1, 256, 256, 1])
                    tf.summary.image('val_recons_gpStrokes', eval_recons_gp_strokes, step=step, max_outputs=5)
                    eval_precontext_strokes = tf.slice(val_pre_context_img, [0, 0, 0, 0], [1, -1, -1, -1])  # [1, 256, 256, 1]
                    tf.summary.image('val_precontext_strokes', eval_precontext_strokes, step=step, max_outputs=1)
                    eval_precontext_recons_strokes = tf.slice(val_pre_context_recons_img, [0, 0, 0, 0], [1, -1, -1, -1])
                    tf.summary.image('val_precontext_recons_strokes', eval_precontext_recons_strokes, step=step, max_outputs=1)
                    tf.summary.image('val_fullStroke', val_full_strokeMap, step=step, max_outputs=4)
                    eval_gp_depth = tf.slice(val_gp_dnmaps, [0, 0, 0, 0, 0], [1, -1, -1, -1, 1])
                    eval_gp_depth = tf.reshape(eval_gp_depth, [-1, 256, 256, 1])
                    tf.summary.image('val_gp_depth', eval_gp_depth, step=step, max_outputs=5)
                    eval_gp_normal = tf.slice(val_gp_dnmaps, [0, 0, 0, 0, 1], [1, -1, -1, -1, 3])
                    eval_gp_normal = tf.reshape(eval_gp_normal, [-1, 256, 256, 3])
                    tf.summary.image('val_gp_normal', eval_gp_normal, step=step, max_outputs=5)

        # save model
        if step % hyper_params['saveModelStep'] == 0 and step > 0:
            ckpt_save_path = ckpt_manager.save()
            train_logger.info('Save model at step: {:d} to file: {}'.format(step, ckpt_save_path))


def test_net():
    # Set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')

    # reader
    readerTest = GPRegTFReader(data_dir=hyper_params['dbDir'], batch_size=1, shuffle=False,
                             raw_size=[hyper_params['imgSize'], hyper_params['imgSize']], infinite=False, prefix='test')

    ckpt = tf.train.Checkpoint(modelAESSG=modelAESSG, transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, hyper_params['ckpt'], max_to_keep=50)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        test_logger.info('restore from the checkpoint {}'.format(ckpt_manager.latest_checkpoint))

    # save model
    tf.saved_model.save(transformer, os.path.join(output_folder, 'gpTFNet'))
    test_logger.info('Write model to gpTFNet')

    tf.saved_model.save(modelAESSG, os.path.join(output_folder, 'embedNet'))
    test_logger.info('Write model to embedNet')

    test_true_acc_metric = tf.keras.metrics.Mean(name='test_true_acc', dtype=tf.float32)
    test_golden_acc_metric = tf.keras.metrics.Mean(name='test_golden_acc', dtype=tf.float32)
    try:
        test_itr = 1
        while True:
            
            input_raw, label_raw, _, _, _, _, _, _, _, nb_gps, nb_strokes, _, _, _ = readerTest.next()

            test_gp_img, test_label, test_token, _ = cook_raw2(modelAESSG,
                                                      input_raw,
                                                      label_raw,
                                                      nb_gps,
                                                      nb_strokes,
                                                      hyper_params['d_model'])

            test_true_acc, test_pred_label = test_iterative_decode_step(test_gp_img, test_label,
                                                                        test_true_acc_metric, input_raw)
            test_golden_acc, test_golden_label = test_golden_decode_step(test_gp_img, test_token, 
                                                                         test_label, test_golden_acc_metric, input_raw)

            if test_itr < 50:
                test_logger.info('\n true_label:\n {}, \n deco_label:\n {}, \n golden label:\n {}'.format(test_label, test_pred_label, test_golden_label))
            test_logger.info('Testing step {}, true acc is: {}, golden acc is: {}'.format(test_itr, test_true_acc, test_golden_acc))

            test_itr += 1
    except StopIteration:
        test_logger.info('Testing average loss is: {}, true acc is: {}, golden acc is: {}'.
                         format(val_loss_metric.result(), test_true_acc_metric.result(), test_golden_acc_metric.result()))


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
