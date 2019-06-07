#!/usr/bin/env python
import tensorflow as tf
from time import time
import numpy as np

def main(optimizer):
  # Model parameters
  W = tf.Variable([.3], dtype=tf.float32) # derivative of of the line y=wx+b
  b = tf.Variable([-.3], dtype=tf.float32)
  # Model input and output
  x = tf.placeholder(tf.float32)
  linear_model = W * x + b
  y = tf.placeholder(tf.float32)

  # loss
  loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
  # optimizer
  #optimizer = tf.train.GradientDescentOptimizer(0.01)
  #optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
  train = optimizer.minimize(loss)

  # training data
  x_train = [1, 2, 3, 4]
  y_train = [0, -1, -2, -3]
  # training loop
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init) # reset values to wrong

  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

  loss_lst = []
  start = time()
  for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    loss_lst.append(curr_loss)
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
  duration = time() - start

  return np.array(loss_lst), duration
