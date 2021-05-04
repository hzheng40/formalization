import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def kl_anneal_function(function, step, k, x0):
    if function == 'logistic':
        return float(1/(1 + np.exp(-k*(step - x0))))
    elif function == 'linear':
        return min(1, step/x0)

batch_size = 128
last_batch_size = 84

# load
data = np.load('data/losses_log/losses_ptb_weightfix3_nodropout_5000crossover.npz')
t_rec = data['t_rec']
t_kl = data['t_kl']
t_elbo = data['t_elbo']

# normalize
t_rec[:, -1] /= last_batch_size
t_kl[:, -1] /= last_batch_size
t_rec[:, :-1] /= batch_size
t_kl[:, :-1] /= batch_size

t_kl_flat = t_kl.flatten()
kl_w = np.array([kl_anneal_function('logistic', step, 0.0025, 10000) for step in range(t_kl_flat.shape[0])])

# df = pd.DataFrame(columns=['epoch', 'train_rec_loss', 'train_kl_loss'])
# for i in range(t_rec.shape[0]):
#     for j in range(t_rec.shape[1]):
#         df = df.append({'epoch': i, 'train_rec_loss': t_rec[i, j], 'train_kl_loss': t_kl[i, j]}, ignore_index=True)

# sns.lineplot(data=df, x='epoch', y='train_rec_loss', ci='sd')
# plt.show()

# sns.lineplot(data=df, x='epoch', y='train_kl_loss', ci='sd')
# plt.show()

plt.plot(t_rec.flatten() + np.multiply(kl_w, t_kl_flat))
plt.show()
plt.plot(t_rec.flatten())
plt.show()

# kl term with weight
fig, ax1 = plt.subplots()
ax1.set_xlabel('step')
ax1.set_ylabel('KL annealed weight')
ax1.plot(kl_w, color='red', label='weight')
ax2 = ax1.twinx()
ax2.set_ylabel('KL term value')
ax2.plot(t_kl_flat, label='raw value')
ax2.plot(np.multiply(kl_w, t_kl_flat), label='weighted value')
fig.legend()
plt.show()