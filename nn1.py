import tensorflow as tf

x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [0.]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([1, 2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

h1 = tf.matmul(X, W1) + b1
h1s = tf.sigmoid(h1)
h2 = tf.matmul(h1s, W2) + b2

hypothesis = tf.sigmoid(h2)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W1), sess.run(W2),
                  sess.run(b1), sess.run(b2))
