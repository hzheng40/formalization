import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# KL
kl_weight_df = pd.read_csv('data/losses_log/run-.-tag-KL Annealed Weight (2).csv')
kl_df = pd.read_csv('data/losses_log/run-.-tag-KL Divergence Weighted (2).csv')

# Cross Entropy
ce_df = pd.read_csv('data/losses_log/run-.-tag-Cross Entropy (2).csv')

# ELBO
elbo_df = pd.read_csv('data/losses_log/run-.-tag-ELBO (2).csv')

# KL plotting
sns.set_context('poster')
ax1 = sns.lineplot(data=kl_df, x='Step', y='Value', label='Weighted KL Term', legend=False)
ax2 = ax1.twinx()
sns.lineplot(data=kl_weight_df, x='Step', y='Value', ax=ax2, color='r', label='KL Annealed Weight', legend=False)
ax1.figure.legend()
plt.show()

# Cross Entropy
ax3 = sns.lineplot(data=ce_df, x='Step', y='Value', label='Cross Entropy Term', legend=False)
ax3.figure.legend()
plt.show()

# ELBO
ax4 = sns.lineplot(data=elbo_df, x='Step', y='Value', label='ELBO', legend=False)
ax4.figure.legend()
plt.show()