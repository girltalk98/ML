
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    W1 = tf.Variable(tf.random_uniform([784, 100], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([100, 100], -1.0, 1.0))
    W3 = tf.Variable(tf.random_uniform([100, 10], -1.0, 1.0))

    b1 = tf.Variable(tf.zeros([100]))
    b2 = tf.Variable(tf.zeros([100]))
    b3 = tf.Variable(tf.zeros([10]))

    hyL1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    hyL2 = tf.nn.relu(tf.matmul(hyL1, W2) + b2)
    y = tf.matmul(hyL2, W3) + b3

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    bef = datetime.now()
    for step in range(5001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if step % 50 == 0:
            print(step, sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    aft = datetime.now()
    calctm = aft-bef

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy = ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print('Learning Time = ', calctm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)