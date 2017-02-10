import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


x_data = [[1., 2., 3.], [5., 2., 8.], [1., 1., 1.]]
y_data = [0., 0., 1.]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.sigmoid(h)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)))

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}),sess.run(W) )

    pltx1 = np.arange(-10, 10, 0.1)
    pltx2 = np.arange(-10, 10, 0.1)

    mX, mY = np.meshgrid(pltx1, pltx2)

    #plty = tf.div(1., 1.+np.exp(-(sess.run(W[0, 0])*pltx1 + sess.run(W[0, 1])*pltx2 + sess.run(W[0, 2]))))
    plty = 1./(1. + np.exp(-(sess.run(W[0, 0]) * mX + sess.run(W[0, 1]) * mY + sess.run(W[0, 2]))))
    #plty = 5


    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    '''''
    surf = ax.plot_surface(pltx1, pltx2, plty,  # data values (2D Arryas)
                           rstride=2,  # row step size
                           cstride=2,  # column step size
                           cmap=cm.RdPu,  # colour map
                           linewidth=1,  # wireframe line width
                           antialiased=True)
                           '''

    surf = ax.plot_surface(mX, mY, plty, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(-1.2, 1.2)
    ax.set_title('Hyperbolic Paraboloid')  # title
    ax.set_xlabel('x label')  # x label
    ax.set_ylabel('y label')  # y label
    ax.set_zlabel('z label')  # z label
    fig.colorbar(surf, shrink=0.5, aspect=5)  # colour bar

    ax.view_init(elev=30, azim=70)  # elevation & angle
    ax.dist = 8  # distance from the plot
    plt.show()



