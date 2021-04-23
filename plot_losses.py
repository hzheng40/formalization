import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/losses_log/losses_ptb_embfix_nodropout.npz')
t_rec = data['t_rec']
t_kl = data['t_kl']
v_rec = data['v_rec']
v_kl = data['v_kl']

df = pd.DataFrame(columns=['epoch', 'train_rec_loss', 'train_kl_loss'])
for i in range(t_rec.shape[0]):
    for j in range(t_rec.shape[1]):
        df = df.append({'epoch': i, 'train_rec_loss': t_rec[i, j], 'train_kl_loss': t_kl[i, j]}, ignore_index=True)

sns.lineplot(data=df, x='epoch', y='train_rec_loss', ci='sd')
plt.show()

sns.lineplot(data=df, x='epoch', y='train_kl_loss', ci='sd')
plt.show()