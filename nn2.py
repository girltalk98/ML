import tensorflow as tf

x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0.], [1.], [1.], [0.]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0), name="Weight1")
W2 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0), name="Weight2")

b1 = tf.Variable(tf.zeros([1, 10]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

h1 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(h1, W2) + b2)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)))

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

tf.summary.scalar('cost', cost)
summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter("./tb", sess.graph)

    for step in range(10001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            summary_str = sess.run(summary, feed_dict={X: x_data, Y: y_data})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W1), sess.run(W2),
                  sess.run(b1), sess.run(b2))

    correctcheck = tf.equal(tf.floor(hypothesis+0.5), Y)
    print(sess.run(correctcheck, feed_dict={X: x_data, Y: y_data}))
