import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multilayer import main
import tensorflow as tf
from collections import OrderedDict
from os import mkdir, path
import pandas as pd
import numpy as np

rand_choices = [2, 3] #[1, 2, 3]
new_per_epoch = [False, True]
batch_sizes = [10, 50, 100]
dos = [0.5]
epochs = 10


output_dir = 'results_multilayer'
try:
  # Create target Directory
  mkdir(output_dir)
except FileExistsError:
  print("Directory " , output_dir ,  " already exists")

for rand_choice in rand_choices:
  for npe in new_per_epoch:
    for batch_size in batch_sizes:
      for do in dos:
        tr_loss, val_loss, tr_acc, val_acc, duration, acc = main(
          rand_choice=rand_choice,
          batch_size=batch_size,
          new_batch_per_epoch=npe,
          do=do)
        #Loss plot
        plt.plot(tr_loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc='upper left')
        plt.savefig(path.join(output_dir, f'loss_{rand_choice}_{batch_size}_{npe}_{do}.png'))
        plt.close()

        #Acc plot
        plt.plot(tr_acc)
        plt.plot(val_acc)
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc='upper left')
        plt.savefig(path.join(output_dir, f'acc_{rand_choice}_{batch_size}_{npe}_{do}.png'))
        plt.close()

        np.save(f'{output_dir}/loss_{rand_choice}_{batch_size}_{npe}_{do}.npy', tr_loss)
        np.save(f'{output_dir}/val_loss_{rand_choice}_{batch_size}_{npe}_{do}.npy', val_loss)
        np.save(f'{output_dir}/acc_{rand_choice}_{batch_size}_{npe}_{do}.npy', tr_acc)
        np.save(f'{output_dir}/val_acc_{rand_choice}_{batch_size}_{npe}_{do}.npy', val_acc)

        dict_results = OrderedDict({
              'Rand ch': rand_choice,
              'Rp': npe,
              'Bs': batch_size,
              'DO': do,
              'Acc': acc,
              'Duration': round(duration,2)
            })
        df_results = pd.DataFrame([dict_results])
        df_results.to_csv(f'{output_dir}/results_exp.csv', mode='a', header=False, index=False)
        # Rand_choice,Rpe,Bs,DO,Acc,Duration
