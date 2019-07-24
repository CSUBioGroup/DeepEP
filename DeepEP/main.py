# encoding=utf-8
import os

import dataloader
import numpy as np
import tensorflow as tf
from protein_cnn import CnnModel
from sklearn import metrics
import pickle

loader = dataloader.DataMaster()

batch_size = 32

epoch_num_cnn = 40

keep_pro = 0.95
init_learning_rate = 0.001

decay_rate = 0.96

decay_steps = loader.training_size / batch_size

model = CnnModel(init_learning_rate, decay_steps, decay_rate)


def validataion():
    # model.prediction_fused
    print('============= begin to test =============')
    step_size = 300
    outputs = []
    logits_pred = []
    for i in range(0, len(loader.test_Y), step_size):
        batch_X = loader.test_X[i:i + step_size]
        batch_E = loader.test_E[i:i + step_size]
        batch_Y = loader.test_Y[i:i + step_size]
        output, y_logit = sess.run([model.prediction_cnn, model.logits_pred],
                                   feed_dict={model.x: batch_X, model.e: batch_E, model.y: batch_Y,
                                              model.dropout_keep_prob: 1.0})
        outputs.append(output)
        logits_pred.append(y_logit)
    y_pred = np.concatenate(outputs, axis=0)
    logits_pred = np.concatenate(logits_pred, axis=0)
    print(">>>> accuracy %.6f" % metrics.accuracy_score(loader.test_Y, y_pred))
    print(">>>> Precision %.6f" % metrics.precision_score(loader.test_Y, y_pred))
    print(">>>> Recall %.6f" % metrics.recall_score(loader.test_Y, y_pred))
    print(">>>> f1_score %.6f" % metrics.f1_score(loader.test_Y, y_pred))
    fpr, tpr, threshold = metrics.roc_curve(loader.test_Y, logits_pred)
    print(">>>> auc_socre %.6f" % metrics.auc(fpr, tpr))
    report = metrics.classification_report(loader.test_Y, y_pred,
                                           target_names=['Trivial', 'Essential'])
    print(report)
    return logits_pred, loader.test_Y


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('pretraining CNN Part')
    for epoch in range(epoch_num_cnn):
        loader.shuffle()
        for iter, idx in enumerate(range(0, loader.training_size, batch_size)):
            batch_X = loader.train_X[idx:idx + batch_size]
            batch_E = loader.train_E[idx:idx + batch_size]
            batch_Y = loader.train_Y[idx:idx + batch_size]
            batch_loss, y_pred, y_logits, accuracy, _ = sess.run(
                [model.loss_cnn, model.prediction_cnn, model.logits_pred, model.accuracy, model.optimizer_cnn],
                feed_dict={model.x: batch_X, model.e: batch_E, model.y: batch_Y,
                           model.dropout_keep_prob: keep_pro})
            if iter % 20 == 0:
                print("=====epoch:%d iter:%d=====" % (epoch + 1, iter + 1))
                print('batch_loss %.3f' % batch_loss)
                print("accuracy %.6f" % metrics.accuracy_score(batch_Y, y_pred))
                print("Precision %.6f" % metrics.precision_score(batch_Y, y_pred))
                print("Recall %.6f" % metrics.recall_score(batch_Y, y_pred))
                print("f1_score %.6f" % metrics.f1_score(batch_Y, y_pred))
                fpr, tpr, threshold = metrics.roc_curve(batch_Y, y_logits)
                print("auc_socre %.6f" % metrics.auc(fpr, tpr))

    eval_res = validataion()

