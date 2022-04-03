# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:14:50 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:51:29 2022

@author: user
"""
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from matplotlib.widgets import Slider


start = time.time()
dataset = pd.read_csv('UPSsample.txt', delimiter='\t', skiprows=4,header=None)

array = dataset.to_numpy()

datax = array[:,0]-5
datay = array[:,2]

#diffy=np.zeros(datax.size-1)
#reducedx=np.zeros(datax.size-1)

#for i in range(datay.size-1):
    #diffy[i]=datay[i]-datay[i+1]
    #reducedx[i]=(datax[i]+datax[i+1])/2

#print(diffy)
c = pd.Series(datax)
d = pd.Series(datay)
x=(c.rolling(100).mean())
yhat=(d.rolling(100).mean())
#yhat = savgol_filter(array[:, 2], 21, 4)

#fig = plt.figure(figsize=(13,15))
#spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1,2])
#main = plt.subplot(spec[1])
#plt.xlabel('Kinetic Energy (eV)')
#plt.ylabel('Electron Intensity (cps)')
#plt.title('Showing the oversmoothing of the UPS spectrum and the average value of the intersection between smoothed data and raw data in red', wrap=True)

#residual = plt.subplot(spec[0])
#s,=residual.plot(array[:, 1],array[:,2]-yhat,'.', color = 'black')
#plt.ylabel('Intensity Difference (cps)')
end = time.time()
print("timE",end - start)
print(np.sum(np.power(array[:,2]-yhat,2)))

axes = plt.gca()
y_min, y_max = axes.get_ylim()
rang= y_max - y_min
print(rang)

plt.plot(x, yhat,'-', label = 'Smoothed curve')
plt.subplots_adjust(bottom=0.25)
plt.plot(datax, array[:, 2],'-', label = 'Other')
plt.plot(4.478423333,1036140,'*', color = 'red')


#defining slider
#ax_slide=plt.axes([0.25, 0.1, 0.65, 0.03])
#properties of slider
#win_len = Slider(ax_slide,'Window length', valmin=5, valmax=51, valinit=51, valstep=2)
#creating update funcyion
#def update(val):
    #current_v = int(win_len.val)
    #new_y = (d.rolling(current_v).mean())
    #new_x = (c.rolling(current_v).mean())
    #p.set_ydata(new_y)
    #s.set_ydata(array[:,2]-new_y)
    #p.set_xdata(new_x)
    #s.set_xdata(array[:,0]-new_x)
    #fig.canvas.draw()
    
#win_len.on_changed(update)

plt.show()

print(array[:,2][26])