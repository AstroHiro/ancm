#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:00:29 2021

@author: hiroyasu
"""

import numpy as np
from classncm import NCM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model,Model
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rc('text',usetex=True)
DPI = 300

this = np.load("data/this.npy")
XXhis = np.load("data/XXhis.npy")
UUhis = np.load("data/UUhis.npy")
pphis = np.load("data/pphis.npy")
ppNNhis = np.load("data/ppNNhis.npy")

n_con = 6
FontSize = 20
FigSize=(20,10)
matplotlib.rcParams.update({"font.size": 15})
matplotlib.rc("text",usetex=True)
tit1 = "Performance of NCM-based Control (1)"
tit2 = "Performance of NCM-based Control (2)"
ly = r"tracking error: $\|x-x_d\|_2$"
l1 = r"tracking error"
l2 = r"optimal steady-state upper bound"
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(14,4))
fig.subplots_adjust(wspace=0.3)
alp = 0.9
ax1.plot(this,np.sqrt(np.sum((XXhis[6,:,:])**2,0)),color="C2",lw=2,alpha=alp)
ax1.plot(this,np.sqrt(np.sum((XXhis[3,:,:])**2,0)),color="C4",lw=2,alpha=alp)
ax1.plot(this,np.sqrt(np.sum((XXhis[2,:,:])**2,0)),color="C1",lw=2,alpha=alp)
ax1.plot(this,np.sqrt(np.sum((XXhis[0,:,:])**2,0)),color="C0",lw=2,alpha=alp)
ax1.plot(this,np.sqrt(np.sum((XXhis[1,:,:])**2,0)),color="C3",lw=2,alpha=alp)
ax1.set_xlabel(r"time [s]",fontsize=FontSize)
ax1.set_ylabel(ly,fontsize=FontSize)
ax1.set_ylim([0,3.5])
ax1.tick_params(labelsize=FontSize)
ax1.grid()
ax1.legend([r"iLQR",r"Feedback linearization",r"aNCM in Theorem 2",r"aNCM in Theorem 3",r"NCM in Theorem 1 (baseline)"],loc="right",bbox_to_anchor=(1.0,0.35))
#plt.legend([l1,l2],loc="best")
ax1.set_title("cart-pole with unknown drags (Sec. V-1)",fontsize=FontSize)

ax2.plot(this,np.sqrt(np.sum((XXhis[7,:,:])**2,0)),color="C2",lw=2,alpha=alp)
ax2.plot(this,np.sqrt(np.sum((XXhis[4,:,:])**2,0)),color="C0",lw=2,alpha=alp)
ax2.plot(this,np.sqrt(np.sum((XXhis[5,:,:])**2,0)),color="C3",lw=2,alpha=alp)
ax2.set_xlabel(r"time [s]",fontsize=FontSize)
ax2.set_ylabel(ly,fontsize=FontSize)
ax2.set_ylim([0,3.5])
ax2.tick_params(labelsize=FontSize)
ax2.grid()
ax2.legend([r"iLQR",r"aNCM for neural nets in Theorem 4",r"NCM in Theorem 1 (baseline)"],loc="right",bbox_to_anchor=(1,0.35))
#plt.legend([l1,l2],loc="best")
ax2.set_title("cart-pole modeld by neural net (Sec. V-2)",fontsize=FontSize)
fname = "figs/cp_ancm_fig.png"
fig.savefig(fname,bbox_inches='tight',dpi=DPI)