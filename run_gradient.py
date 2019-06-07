import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradient_descent import main
import tensorflow as tf
from collections import OrderedDict
from os import mkdir, path
import pandas as pd
import numpy as np

optimizer_names = ['gd', 'adam', 'rmsprop']
lrs = [0.1, 0.01, 0.001]

output_dir = 'results_gradient'
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

    loss, duration = main(optimizer)
    #Loss plot
    plt.plot(loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(path.join(output_dir, f'loss_{optimizer_name}_{lr}.png'))
    plt.close()
    np.save(f'{output_dir}/loss_{optimizer_name}_{lr}.npy', loss)

    print(loss.shape)
    dict_results = OrderedDict({
          'Optimizer': optimizer_name,
          'LR': lr,
          'Loss': round(loss[-1], 3) if isinstance(loss[-1],float) else loss[-1],
          'Duration': round(duration,2)
        })
    df_results = pd.DataFrame([dict_results])
    df_results.to_csv(f'{output_dir}/results_exp.csv', mode='a', header=False, index=False)

  