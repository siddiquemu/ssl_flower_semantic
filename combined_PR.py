import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from cmcrameri import cm
from cmcrameri import show_cmaps
from cmcrameri.cm import paths
import numpy as np

#show_cmaps()
print(paths)
#print(cm._cmap_base_names_categorical)
colors = np.genfromtxt('/home/siddique/miniconda3/envs/det2/lib/python3.7/site-packages/cmcrameri/cmaps/batlowS.txt')
colorsPear = np.genfromtxt('/home/siddique/miniconda3/envs/det2/lib/python3.7/site-packages/cmcrameri/cmaps/actonS.txt')
#print(colors.shape)
files_ssl = glob.glob('./PR_ablation/CV1/*SSL.csv')
files_ssl_rgr = glob.glob('./PR_ablation/CV1/*SSL_RGR.csv')
color = {'AppleA':'red', 'AppleB':'green', 'Peach':'blue', 'Pear':'magenta'}
color = {'AppleA':colors[10], 'AppleB':colors[12], 'Peach':colors[45], 'Pear':colorsPear[80]}
print(color)
for d_name in color.keys():

    for file in files_ssl:
        if d_name in file.split('_'):
            file_ssl = file

    for file in files_ssl_rgr:
        if d_name in file.split('_'):
            file_ssl_rgr = file

    print(file_ssl, file_ssl_rgr)
    # plot ssl model eval
    data = pd.read_csv(file_ssl)
    #print(data.head())
    max_val = data[data['F1']==data['F1'].max()]
    print(max_val['F1'].values)
    
    print(plt.style.available)
    plt.style.use('tableau-colorblind10')
    
    plt.figure(1, figsize=(5, 3.5))


    plt.plot(data['recall'].values, data['precision'].values, c=color[d_name],linestyle='dashed', label=f'{d_name}: SSL')
    plt.scatter(x=max_val['recall'].values[0], y=max_val['precision'].values[0], c=color[d_name])
    plt.legend()

    #plot ssl_rgr model eval
    #d_name = file_ssl_rgr.split('_')[1]

    data = pd.read_csv(file_ssl_rgr)
    #print(data.head())
    max_val = data[data['F1']==data['F1'].max()]
    print(max_val['F1'].values)
    print('total data points', len(data['recall'].values))
    plt.plot(data['recall'].values, data['precision'].values, c=color[d_name], label=f'{d_name}: SSL+RGR')
    plt.scatter(x=max_val['recall'].values[0], y=max_val['precision'].values[0], c=color[d_name])
    plt.legend()

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
#plt.xticks(np.arange(0.02, 1.0, 0.02))
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.tight_layout()
plt.savefig(f'PR_combined.eps', dpi=600, bbox_inches='tight')
plt.savefig(f'PR_combined.png', dpi=200, bbox_inches='tight')

