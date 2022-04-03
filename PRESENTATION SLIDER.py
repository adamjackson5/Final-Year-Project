# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:51:29 2022

@author: user
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from matplotlib.widgets import Slider


dataset = pd.read_csv('UPSsample.txt', delimiter='\t', skiprows=4,header=None)

array = dataset.to_numpy()

datax = array[:,0]
x= array[:,0]

datay = array[:,2]

diffy=np.zeros(datax.size-1)
reducedx=np.zeros(datax.size-1)

for i in range(datay.size-1):
    diffy[i]=datay[i]-datay[i+1]
    reducedx[i]=(datax[i]+datax[i+1])/2

print(diffy)

#makes usure loading even number of data points all over - stick to odd for moving av
c = pd.Series(datax)
d = pd.Series(datay)
#x=(c.rolling(17).mean())
yhat = savgol_filter(array[:, 2], 23, 4)
fig = plt.figure(figsize=(13,15))
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[1,2])
main = plt.subplot(spec[1])
plt.xlabel('Binding Energy (eV)')
plt.ylabel('Electron Intensity (cps)')
residual = plt.subplot(spec[0])
s,=residual.plot(array[:, 1],array[:,2]-yhat,'.', color = 'black')
plt.ylabel('Difference in Electron Intensity (cps)')


plt.ylim(-1000,1000)
plt.title('Savgol filter at window 23 and order of 4 applied to UPS spectrum wuth derivative plotted', wrap=True)

print(np.sum(np.power(array[:,2]-yhat,2)))

axes = plt.gca()
y_min, y_max = axes.get_ylim()
rang= y_max - y_min
print(rang)


#p,=main.plot(x, yhat,'-', label = 'Smoothed curve', color = 'orange')
p,=main.plot(x, yhat,'-', label = 'Smoothed curve')

plt.subplots_adjust(bottom=0.25)
main.plot(x, array[:, 2],'.', label = 'Other')
main.plot(x, array[:, 2],'-', label = 'Other')

main.plot(reducedx, diffy,'*', label = 'Other', color = 'green')




#defining slider
ax_slide=plt.axes([0.25, 0.1, 0.65, 0.03])
#properties of slider
win_len = Slider(ax_slide,'Window length', valmin=5, valmax=51, valinit=51, valstep=2)
#creating update funcyion
def update(val):
    current_v = int(win_len.val)
    new_y = savgol_filter(array[:, 2], current_v, 4)
    p.set_ydata(new_y)
    s.set_ydata(array[:,2]-new_y)
    fig.canvas.draw()
    
win_len.on_changed(update)

plt.show()
