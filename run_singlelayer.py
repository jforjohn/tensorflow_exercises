import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from singlelayer import main
import tensorflow as tf
from collections import OrderedDict
from os import mkdir, path
import pandas as pd
import numpy as np

optimizers = [tf.train.GradientDescentOptimizer(0.01), tf.train.AdamOptimizer(learning_rate=0.1)]
optimizer_names = ['gd', 'adam', 'rmsprop']
lrs = [0.1, 0.01, 0.001, 0.0001, 0.00001]

output_dir = 'results_singlelayer'
try:
  # Create target Directory
  mkdir(output_dir)
except FileExistsError:
  print("Directory " , output_dir ,  " already exists")

for optimizer_name in optimizer_names:
  for lr in lrs:
    if optimizer_name == 'gd':
      optimizer = tf.train.GradientDescentOptimizer(lr)

    elif optimizer_name == 'adam':
      optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    elif optimizer_name == 'rmsprop':
      optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)

    tr_loss, val_loss, tr_acc, val_acc, duration, acc = main(optimizer)
    #Loss plot
    plt.plot(tr_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(path.join(output_dir, f'loss_{optimizer_name}_{lr}.png'))
    plt.close()

    #Acc plot
    plt.plot(tr_acc)
    plt.plot(val_acc)
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(path.join(output_dir, f'acc_{optimizer_name}_{lr}.png'))
    plt.close()

    np.save(f'{output_dir}/loss_{optimizer_name}_{lr}.npy', tr_loss)
    np.save(f'{output_dir}/val_loss_{optimizer_name}_{lr}.npy', val_loss)
    np.save(f'{output_dir}/acc_{optimizer_name}_{lr}.npy', tr_acc)
    np.save(f'{output_dir}/val_acc_{optimizer_name}_{lr}.npy', val_acc)

    dict_results = OrderedDict({
          'Optimizer': optimizer_name,
          'LR': lr,
          'Acc': acc,
          'Duration': round(duration,2)
        })
    df_results = pd.DataFrame([dict_results])
    df_results.to_csv(f'{output_dir}/results_exp.csv', mode='a', header=False, index=False)

  