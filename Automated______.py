
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:51:29 2022

@author: user
"""
import time
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np

start = time.time()
dataset = pd.read_csv('UPSsample.txt', delimiter='\t', skiprows=4,header=None)

array = dataset.to_numpy()

datax = array[:,0]
datay = array[:,2]

diffy=np.zeros(datax.size-1)
reducedx=np.zeros(datax.size-1)
yhat = savgol_filter(array[:, 2], 51, 4)
for i in range(yhat.size-1):
    diffy[i]=datay[i]-datay[i+1]
    reducedx[i]=(datax[i]+datax[i+1])/2
min_value = min(diffy)

min_index = np.where(diffy == min_value)


print("x",reducedx[min_index]-5)
end = time.time()
print(end - start)