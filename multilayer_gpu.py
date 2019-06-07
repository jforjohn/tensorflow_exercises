#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import sys
import random
from time import time
from tensorflow.python.client import device_lib


def check_available_gpus():
  local_devices = device_lib.list_local_devices()
  gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
  gpu_num = len(gpu_names)

  print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

  return gpu_names

def get_model(reuse, x, is_training, keep_prob):
  # Define a scope for reusing the variables
  with tf.variable_scope('MNIST', reuse=reuse):
    #declare weights and biases
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)


    #convolution and pooling
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    #First convolutional layer: 32 features per each 5x5 patch
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])


    #Reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height.
    #28x28 = 784
    #The final dimension corresponding to the number of color channels.
    x_image = tf.reshape(x, [-1, 28, 28, 1])


    #We convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. 
    #The max_pool_2x2 method will reduce the image size to 14x14.

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    #Second convolutional layer: 64 features for each 5x5 patch.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    #Densely connected layer: Processes the 64 7x7 images with 1024 neurons
    #Reshape the tensor from the pooling layer into a batch of vectors, 
    #multiply by a weight matrix, add a bias, and apply a ReLU.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #if is_training:
    #drop_out
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #else:
    #  h_fc1_drop = h_fc1


    #Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv

def main(epochs=10,
         do=0.5,
         batch_size=50,
         gpu_num=1):

  gpu_names = check_available_gpus()
  #gpu_num = len(gpu_names)
  print()
  print('# of GPUs:', gpu_num)
  print('GPU devices:', gpu_names)
  print()

  val_batch_size = 100
  #read data from file
  data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
  #FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
  data = data_input[0]
  batch_num = len(data[0][0])//(batch_size*gpu_num)
  val_batch_num = len(data[1][0])//(val_batch_size*gpu_num)

  new_tr_ind = random.sample(range(len(data[0][0])), len(data[0][0]))
  new_val_ind = random.sample(range(len(data[1][0])), len(data[1][0]))
  #print ( N.shape(data[0][0])[0] )
  #print ( N.shape(data[0][1])[0] )


  #data layout changes since output should an array of 10 with probabilities
  real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
  for i in range ( N.shape(data[0][1])[0] ):
    real_output[i][data[0][1][i]] = 1.0  

  val_output = N.zeros( (N.shape(data[1][1])[0] , 10), dtype=N.float )
  for i in range ( N.shape(data[1][1])[0] ):
    val_output[i][data[1][1][i]] = 1.0

  #data layout changes since output should an array of 10 with probabilities
  real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
  for i in range ( N.shape(data[2][1])[0] ):
    real_check[i][data[2][1][i]] = 1.0

  new_X_data = data[0][0][new_tr_ind]
  new_y_data = real_output[new_tr_ind]

  new_valX_data = data[1][0][new_val_ind]
  new_valy_data = val_output[new_val_ind]
  

  # Place all ops on CPU by default
  with tf.device('/cpu:0'):
      tower_losses = []
      reuse_vars = False

      # tf Graph input
      #set up the computation. Definition of the variables.
      x = tf.placeholder(tf.float32, [None, 784])
      #W = tf.Variable(tf.zeros([784, 10]))
      y_ = tf.placeholder(tf.float32, [None, 10])
      keep_prob = tf.placeholder(tf.float32)

      # Loop over all GPUs and construct their own computation graph
      for i in range(gpu_num):
        with tf.device('/gpu:{}'.format(i)):
          _batch_ini = batch_size*i
          _batch_end = batch_size*i + batch_size
          
          _batch_xs = x[_batch_ini:_batch_end]
          _batch_ys = y_[_batch_ini:_batch_end]

          y_conv = get_model(reuse_vars, _batch_xs, True, keep_prob)

          print()
          print(_batch_ys.get_shape(), y_conv.get_shape())
          #Crossentropy
          # Define loss and optimizer (with train logits, for dropout to take effect)
          #cross_entropy = tf.reduce_mean(
          #   tf.nn.softmax_cross_entropy_with_logits(labels=_batch_ys, logits=y_conv))

          cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=_batch_ys, logits=y_conv)
          # Only first GPU compute accuracy
          if i == 0:
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(_batch_ys, 1))

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

          reuse_vars = True
          tower_losses.append(cross_entropy)

  loss = tf.reduce_mean(tf.concat(tower_losses, axis=0))
    
  train_step = tf.train.AdamOptimizer(0.001).minimize(loss, colocate_gradients_with_ops=True)

  print('Exp:', batch_size, do)
  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    #TRAIN 
    print("TRAINING")

    loss_lst = []
    val_loss_lst = []
    tr_acc_lst = []
    val_acc_lst = []
    start = time()
    for epoch in range(epochs):
      loss_batch = 0
      acc_batch = 0
      for i in range(batch_num):
        #until 1000 96,35%
        batch_ini = batch_size*i*gpu_num
        batch_end = batch_size*i*gpu_num + batch_size*gpu_num
        
        batch_xs = new_X_data[batch_ini:batch_end]
        #data[0][0][batch_ini:batch_end]
        batch_ys = new_y_data[batch_ini:batch_end]
        #real_output[batch_ini:batch_end]

        '''
        if i % 10 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch_xs, y_: batch_ys, keep_prob: 1})
          print('step %d, training accuracy %g Batch [%d,%d]' % (i, train_accuracy, batch_ini, batch_end))
        '''
        
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: do})
        train_accuracy = accuracy.eval(feed_dict={
              x: batch_xs, y_: batch_ys, keep_prob: 1})
        curr_loss = sess.run(loss, {x: batch_xs, y_: batch_ys, keep_prob: do})
        loss_batch += curr_loss
        acc_batch += train_accuracy
      
      loss_lst.append(loss_batch/batch_num)
      tr_acc_lst.append(acc_batch/batch_num)
    
      loss_batch = 0
      acc_batch = 0
      for i in range(val_batch_num):
        val_batch_ini = val_batch_size*i*gpu_num
        val_batch_end = val_batch_size*i*gpu_num + val_batch_size*gpu_num
        batch_val_xs = new_valX_data[val_batch_ini:val_batch_end]
        batch_val_ys = new_valy_data[val_batch_ini:val_batch_end]
        val_loss = sess.run(loss, {x: batch_val_xs, y_: batch_val_ys, keep_prob: do})
        #val_acc = accuracy.eval(feed_dict={x: batch_val_xs, y_: batch_val_ys, keep_prob: 1})
        val_acc = sess.run(accuracy,
                  feed_dict={x: batch_val_xs,
                            y_: batch_val_ys,
                            keep_prob: 1})
        loss_batch += val_loss
        acc_batch += val_acc

      val_loss_lst.append(loss_batch/val_batch_num)
      val_acc_lst.append(acc_batch/val_batch_num)

    percent = 100.0 * epoch / epochs
    line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
    status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} loss:{4:.2f} val_loss:{5:.2f} acc:{6:.2f} val_acc:{7:.2f}'
    sys.stdout.write(status.format(percent, line, epoch, epochs, loss_lst[-1], val_loss_lst[-1], tr_acc_lst[-1], val_acc_lst[-1]))
    duration = time() - start
    print()
    print('Training duration:', duration)
    print()

    #TEST
    print("TESTING")

    #tf_acc = accuracy.eval(feed_dict={x: data[2][0], y_: real_check, keep_prob: 1})
    tf_acc = N.mean([sess.run(accuracy, feed_dict={x: data[2][0][i:i+batch_size],
      y_: real_check[i:i+batch_size], keep_prob: 1}) for i in range(0, len(data[2][0]), batch_size)])
    print('test accuracy %g' %(tf_acc))
  
  return N.array(loss_lst), N.array(val_loss_lst), N.array(tr_acc_lst), N.array(val_acc_lst), duration, tf_acc
