import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multilayer_gpu import main
import tensorflow as tf
from collections import OrderedDict
from os import mkdir, path
import pandas as pd
import numpy as np
import sys

epochs = 10
do = 0.5
batch_size = 50
if len(sys.argv) == 2:
  try:
    gpu_num = int(sys.argv[1])
  except ValueError:
    print('Argument: number of GPUs: int')
    sys.exit(1)
else:
  sys.exit(1)

output_dir = 'results_multilayer_gpu'
try:
  # Create target Directory
  mkdir(output_dir)
except FileExistsError:
  print("Directory " , output_dir ,  " already exists")

tr_loss, val_loss, tr_acc, val_acc, duration, acc= main(
  batch_size=batch_size,
  do=do,
  gpu_num=gpu_num)
#Loss plot
plt.plot(tr_loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(path.join(output_dir, f'loss_{batch_size}_{do}_{gpu_num}.png'))
plt.close()

#Acc plot
plt.plot(tr_acc)
plt.plot(val_acc)
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(path.join(output_dir, f'acc_{batch_size}_{do}_{gpu_num}.png'))
plt.close()

np.save(f'{output_dir}/loss_{batch_size}_{do}_{gpu_num}.npy', tr_loss)
np.save(f'{output_dir}/val_loss_{batch_size}_{do}_{gpu_num}.npy', val_loss)
np.save(f'{output_dir}/acc_{batch_size}_{do}_{gpu_num}.npy', tr_acc)
np.save(f'{output_dir}/val_acc_{batch_size}_{do}_{gpu_num}.npy', val_acc)

dict_results = OrderedDict({
      '#GPU': gpu_num,
      'Bs': batch_size,
      'DO': do,
      'Acc': acc,
      'Duration': round(duration,2)
    })
df_results = pd.DataFrame([dict_results])
df_results.to_csv(f'{output_dir}/results_exp.csv', mode='a', header=False, index=False)
# Rand_choice,Rpe,Bs,DO,Acc,Duration
