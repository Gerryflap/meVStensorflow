import tensorflow as tf
import numpy as np




x_data = np.matrix([[0,0,1,1], [0,1,0,1]]).T.astype("float32")
y_data = np.matrix([[0], [1], [1], [0]]).astype("float32")
print(x_data)


W = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0, dtype=tf.float32))
W2 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0, dtype=tf.float32))
b = tf.Variable(tf.zeros([5]))
y = tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(x_data, W) + b), W2))
print("Y:", tf.cast(y, tf.float32))

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square((y - y_data)))
optimizer = tf.train.GradientDescentOptimizer(0.7)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(4000):
    sess.run(train)
    if step % 1000 == 0:
        print (step, sess.run(W),"\n", sess.run(b))

cp = y-y_data
accuracy = tf.reduce_mean(tf.cast(cp, tf.float32))
print(sess.run(y), y_data)
