
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from cmcrameri import show_cmaps
from cmcrameri.cm import paths

colors = np.genfromtxt('/home/siddique/miniconda3/envs/det2/lib/python3.7/site-packages/cmcrameri/cmaps/batlowS.txt')
colors_buda = np.genfromtxt('/home/siddique/miniconda3/envs/det2/lib/python3.7/site-packages/cmcrameri/cmaps/budaS.txt')
print(colors_buda*255)
colorsPear = np.genfromtxt('/home/siddique/miniconda3/envs/det2/lib/python3.7/site-packages/cmcrameri/cmaps/actonS.txt')
color = {'AppleA':'red', 'AppleB':'green', 'Peach':'blue', 'Pear':'magenta'}
color = {'AppleA':colors[10], 'AppleB':colors[12], 'Peach':colors[45], 'Pear':colorsPear[80]}

lambda_s = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#IoU_Peach_full_set = [58.1, 58.8, 59.3, 59.2, 59.6, 59.8, 60.1, 60.1, 59.8]
IoU_Peach = np.array([0.5938881, 0.60532135, 0.6163796, 0.61395043, 0.62019426, 0.621618, 0.6284386, 0.6307373, 0.6306491])*100
#IoU_Pear_full_set = [75.5, 75.5, 75.3, 75.2, 75.4, 75.4, 75.2, 75.3, 75.2]
IoU_Pear = np.array([0.78099865, 0.7837382, 0.7856335, 0.7862346, 0.7875606, 0.7867461, 0.7879472, 0.7901, 0.7912294])*100
# IoU_AppleB1 = [75.4, 76.2, 76.2, 75.4, 74.8,75.4, 74.4, 75.0, 75.7]
# IoU_AppleB2 = [76.0, 74.8, 74.4, 75.9, 74.7, 75.1, 76.1, 75.8, 75.3]
# IoU_AppleB3 = np.array([0.75341773, 0.7512415, 0.7542879, 0.75461704, 0.7480595, 0.7528726, 0.7522689, 0.7430104, 0.75597996])*100
# IoU_AppleB = (np.array(IoU_AppleB1) + np.array(IoU_AppleB2) + IoU_AppleB2)/3
IoU_AppleB = np.array([0.77569985, 0.7769137, 0.7775972, 0.7781326, 0.7797642, 0.77858764, 0.7788936, 0.777, 0.7765901])*100
#IoU_AppleA_full_set = [73.9, 73.8, 74.2, 74.3, 74.6, 74.6, 74.6, 74.9, 75.0]
IoU_AppleA = [0.749312, 0.75127494, 0.7522898, 0.7545473, 0.7552182, 0.755675, 0.755847, 0.7585, 0.756487]
marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                    markersize=15, markerfacecoloralt='tab:red')
plt.figure(1, figsize = (6, 3.5))
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.plot(lambda_s, IoU_AppleA, c=color['AppleA'],marker='^', markersize=10, linewidth=2, label='AppleA')
plt.plot(lambda_s, IoU_AppleB, c=color['AppleB'],marker='*', markersize=10, linewidth=2, label='AppleB')
plt.plot(lambda_s, IoU_Peach, c=color['Peach'],marker='o', markersize=10, linewidth=2, label='Peach')
plt.plot(lambda_s, IoU_Pear, c=color['Pear'],marker='>', markersize=10, linewidth=2, label='Pear')
plt.legend()
plt.xlabel('$\lambda$', fontsize=15)
plt.xticks(np.arange(0.1, 1, step=0.1))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('IoU', fontsize=15)
plt.grid()
plt.tight_layout()
plt.savefig(f'lambda_rotation_ablation.eps', dpi=200, bbox_inches='tight')
plt.savefig(f'lambda_rotation_ablation.png', dpi=200, bbox_inches='tight')
