#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import sys
from time import time

def main(optimizer, epochs=5):
  #read data from file
  data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
  #FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
  data = data_input[0]
  #print ( N.shape(data[0][0])[0] )
  #print ( N.shape(data[0][1])[0] )
  print(N.array(data)[0][0].shape)
  print(N.array(data)[1][0].shape)
  print(N.array(data)[2][0].shape)

  #data layout changes since output should an array of 10 with probabilities
  real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
  for i in range ( N.shape(data[0][1])[0] ):
    real_output[i][data[0][1][i]] = 1.0  

  val_output = N.zeros( (N.shape(data[1][1])[0] , 10), dtype=N.float )
  for i in range ( N.shape(data[1][1])[0] ):
    val_output[i][data[1][1][i]] = 1.0

  #data layout changes since output should be an array of 10 with probabilities
  real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
  for i in range ( N.shape(data[2][1])[0] ):
    real_check[i][data[2][1][i]] = 1.0

  #set up the computation. Definition of the variables.
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

  # tf.train.GradientDescentOptimizer(0.5)
  train_step = optimizer.minimize(cross_entropy)

  #sess = tf.InteractiveSession()

  #TRAINING PHASE
  print("TRAINING")

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss_lst = []
    val_loss_lst = []
    tr_acc_lst = []
    val_acc_lst = []
    start = time()
    for epoch in range(epochs):
      loss_batch = 0
      acc_batch = 0
      for i in range(500):
        batch_xs = data[0][0][100*i:100*i+100]
        batch_ys = real_output[100*i:100*i+100]

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        curr_loss = sess.run(cross_entropy, {x: batch_xs, y_: batch_ys})
        tr_acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        loss_batch += curr_loss
        acc_batch += tr_acc

      loss_lst.append(loss_batch/500)
      tr_acc_lst.append(acc_batch/500)
      
      loss_batch = 0
      acc_batch = 0
      for i in range(100):
        batch_val_xs = data[1][0][100*i:100*i+100]
        batch_val_ys = val_output[100*i:100*i+100]
        val_loss = sess.run(cross_entropy, {x: batch_val_xs, y_: batch_val_ys})
        val_acc = sess.run(accuracy, feed_dict={x: batch_val_xs, y_: batch_val_ys})
        loss_batch += val_loss
        acc_batch += val_acc

      val_loss_lst.append(loss_batch/100)
      val_acc_lst.append(acc_batch/100)

      percent = 100.0 * epoch / epochs
      line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
      status = '\r{0:3.0f}%{1} {2:3d}/{3:3d}'
      sys.stdout.write(status.format(percent, line, epoch, epochs))
      sys.stdout.flush()

    duration = time() - start

    #CHECKING THE ERROR
    print("ERROR CHECK")

    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf_acc = accuracy.eval(feed_dict={x: data[2][0], y_: real_check})

  return N.array(loss_lst), N.array(val_loss_lst), N.array(tr_acc_lst), N.array(val_acc_lst), duration, tf_acc


