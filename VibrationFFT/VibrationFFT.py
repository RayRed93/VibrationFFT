
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal
os.getcwd()
print("yo")

df = pd.read_excel(io="C:\\Users\\piotr\\source\\repos\\VibrationFFT\\VibrationFFT\\data.xlsx", sheet_name="Data1")
#df = pd.read_excel(io=sys.argv[0], sheet_name="Data1")

#df.plot()
#%%
time = df['Time'].values
amp = df['Amplituda'].values

plt.figure(1, figsize=(13, 4))
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.plot(amp)
#sigGraph.show()

sp = np.fft.fft(amp)

dt = np.mean(time[1:] - time[:-1])
freqs = np.fft.fftfreq(len(sp), dt)
#print(freqs)
mag = np.abs(sp.real)
#mag = sp.real


max_idxs = signal.argrelmax(mag, order=500)[0]
print(max_idxs)
#min_idxs = signal.argrelmin(mag, order=500)[0]


plt.figure(2, figsize=(13, 4))
plt.xlim((0, 30))
plt.ylabel('Magnitude')
plt.xlabel('Frequency')
plt.plot(freqs, mag, freqs[max_idxs], mag[max_idxs], 'ro')
plt.show()
#raw_input()
#%


#print(df.head(65))
