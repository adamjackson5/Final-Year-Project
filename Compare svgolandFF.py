# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:04:06 2022

@author: user
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from matplotlib.widgets import Slider
from scipy.fftpack import fft
from scipy.fftpack import ifft

dataset = pd.read_csv('sample.txt', delimiter='\t', skiprows=4,header=None)

array = dataset.to_numpy()

datax = array[:,1]
datay = array[:,2]

diffy=np.zeros(datax.size-1)
reducedx=np.zeros(datax.size-1)

for i in range(datay.size-1):
    diffy[i]=datay[i]-datay[i+1]
    reducedx[i]=(datax[i]+datax[i+1])/2

print(diffy)

yhat = savgol_filter(array[:, 2], 21, 4)

rft = np.fft.rfft(datay)
rft[12:] = 0   # Note, rft.shape = 21
yff = np.fft.irfft(rft)

#plt.plot(array[:, 1], yhat,'-', label = 'Smooth curve')
plt.plot(array[0:170, 1], yff,'-', label = 'Smooth curve')

plt.xlabel('Binding Energy (eV)')
plt.ylabel('Electron Intensity (cps)')





plt.subplots_adjust(bottom=0.25)
plt.plot(array[:, 1], array[:, 2],'-', label = 'Other')
