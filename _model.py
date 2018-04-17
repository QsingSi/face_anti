import tensorflow as tf
import numpy as np

__all__ = ['model']


class model(object):
    def __init__(self):
        pass

    def patch_cnn(self, input_, labels, mode, params):
        output = input_
        output = tf.layers.conv2d(
            output, filters=50, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, padding='same')
        output = tf.contrib.layers.batch_norm(
            output, scale=True, updates_collections=None)
        output = tf.layers.max_pooling2d(
            output, pool_size=(2, 2), strides=2, padding='same')
        print(output.shape)
        # Conv-2
        output = tf.layers.conv2d(
            output, filters=100, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding='same')
        output = tf.layers.batch_normalization(output, axis=1)
        output = tf.layers.max_pooling2d(
            output, pool_size=(2, 2), strides=2, padding='same')
        print(output.shape)
        # Conv-3
        output = tf.layers.conv2d(
            output, filters=150, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding='same')
        output = tf.layers.batch_normalization(output, axis=1)
        output = tf.layers.max_pooling2d(
            output, pool_size=(3, 3), strides=2, padding='same')
        print(output.shape)
        # Conv-4
        output = tf.layers.conv2d(
            output, filters=200, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding='same')
        output = tf.layers.batch_normalization(output, axis=1)
        output = tf.layers.max_pooling2d(
            output, pool_size=(2, 2), strides=2, padding='same')
        print(output.shape)
        # Conv-5
        output = tf.layers.conv2d(
            output, filters=250, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding='same')
        output = tf.layers.batch_normalization(output, axis=1)
        output = tf.layers.max_pooling2d(
            output, pool_size=(2, 2), strides=2, padding='same')
        print(output.shape)
        # full Connect
        output = tf.layers.conv2d(output, filters=1000, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='valid')
        #weights = np.random.rand(1, 3, 3, 1)
        # output = tf.layers.dense(
        # output, units=1000, kernel_initializer=tf.constant_initializer(weights, tf.float32))
        output = tf.layers.batch_normalization(output, axis=1)
        output = tf.layers.dropout(output, rate=0.5)
        print(output.shape)
        # full Connect-2
        output = tf.layers.dense(output, units=400)
        output = tf.layers.batch_normalization(output, axis=1)
        print(output.shape)
        # full Connect-3
        output = tf.layers.dense(output, units=2)
        print(output.shape)
        return output

    def patch_model(self, features, labels, mode, params):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001, momentum=0.99)
        logits = self.patch_cnn(features, labels, mode, params)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        optimizer.minimize(loss, global_step=tf.train.get_global_step())
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes, name='acc_op')
        metrics = {'accruay': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    def patch_classifier(self, features, labels, params=None):
        classifier = tf.estimator.Estimator(
            model_fn=self.patch_model, params=params)
        # Train the model
        classifier.train()

    def depth_cnn(self, input_):
        output = input_
        # Conv-1
        output = tf.layers.conv2d(output, filters=64, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.layers.conv2d(output, filters=64, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.layers.conv2d(output, filters=128, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.layers.max_pooling2d(
            output, pool_size=(2, 2), strides=2, padding='same')
        # Conv-2
        output = tf.layers.conv2d(output, filters=128, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.layers.conv2d(output, filters=256, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.layers.conv2d(output, filters=160, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.layers.max_pooling2d(
            output, pool_size=(2, 2), strides=2, padding='same')
        # Conv-3
        output = tf.layers.conv2d(output, filters=128, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        sample_num = output.get_shape()[0].value
        output = tf.nn.conv2d_transpose(output, filter=tf.ones([6, 6, 128, 128]), output_shape=[
                                        sample_num, 37, 37, 128], strides=[1, 5, 5, 1], padding='SAME')
        # Conv-4
        output = tf.layers.conv2d(output, filters=128, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.nn.conv2d_transpose(output, filter=tf.ones([6, 6, 128, 128]), output_shape=[
                                        sample_num, 42, 42, 128], strides=[1, 5, 5, 1], padding='SAME')
        # Conv-5
        output = tf.layers.conv2d(output, filters=160, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.nn.conv2d_transpose(output, filter=tf.ones([6, 6, 160, 160]), output_shape=[
                                        sample_num, 47, 47, 160], strides=[1, 5, 5, 1], padding='SAME')
        # Conv-6
        output = tf.layers.conv2d(output, filters=320, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='same')
        output = tf.nn.conv2d_transpose(output, filter=tf.ones([6, 6, 320, 320]), output_shape=[
                                        sample_num, 52, 52, 320], strides=[1, 5, 5, 1], padding='SAME')
        # Conv-7
        output = tf.layers.conv2d(output, filters=1, kernel_size=(
            3, 3), strides=1, activation=tf.nn.relu, padding='SAME')
        return output
