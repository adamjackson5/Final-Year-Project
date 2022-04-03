import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.gridspec as gridspec
import numpy as np


dataset = pd.read_csv('sample.txt', delimiter='\t', skiprows=4,header=None)
  
array = dataset.to_numpy()
print(array)
yhat = savgol_filter(array[:, 2], 31, 3)
#xhat = savgol_filter(array[:, 1], 13, 3)
plt.plot(array[:, 1], yhat,'-', label = 'File Data')
plt.plot(array[:, 1], array[:, 2],'.', label = 'File Data')
plt.xlabel('Binding Energy (eV)')
plt.ylabel('Electron Intensity (cps)')
plt.title('XPS data smoothed by Savgol filter with a window size of 31 and a cubic polynomial', wrap=True)

plt.show()


