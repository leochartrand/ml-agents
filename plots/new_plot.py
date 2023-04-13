#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:21:52 2020

@author: czh513
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

SAVE_DIR = Path("./figures")

def smooth(csv_path, weight=0.8):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step','Value'], dtype={'Step':np.int, 'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step':data['Step'].values, 'Value':smoothed})
    save.to_csv('smooth_' + csv_path)


def smooth_and_plot(csv_path, weight=0.8):
    data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step','Value'], dtype={'Step':np.int, 'Value':np.float})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    steps = data['Step'].values
    steps = steps.tolist()
  
    endrewards=smoothed[180:200]
    
    return endrewards, steps, smoothed

do_minimum = False
endrewards1, steps1, smoothed1 =smooth_and_plot('data/ppo-new.csv')
endrewards2, steps2, smoothed2 =smooth_and_plot('data/poca.csv')
endrewards3, steps3, smoothed3 =smooth_and_plot('data/a2c-ext.csv')
endrewards5, steps5, smoothed5 =smooth_and_plot('data/mpo-inter.csv')

if do_minimum:
    minimum = min(len(steps1), len(steps2), len(steps3), len(steps5))
    steps1 = steps1[:minimum]
    steps2 = steps2[:minimum]
    steps3 = steps3[:minimum]
    steps5 = steps5[:minimum]
    smoothed1 = smoothed1[:minimum]
    smoothed2 = smoothed2[:minimum]
    smoothed3 = smoothed3[:minimum]
    smoothed5 = smoothed5[:minimum]

    if minimum < 200:
        endrewards1 = smoothed1[-20:]
        endrewards2 = smoothed2[-20:]
        endrewards3 = smoothed3[-20:]
        endrewards5 = smoothed5[-20:]


fig = plt.figure(2)
plt.plot(steps1, smoothed1, label='PPO')
plt.plot(steps2, smoothed2, label='MA-POCA')
plt.plot(steps3, smoothed3, label='A2C')
plt.plot(steps5, smoothed5, label='MPO', color="purple", linewidth=2.0)
plt.ylim(min(min(smoothed1), min(smoothed2), min(smoothed3), min(smoothed5)) -10, 1350)
plt.xlim(0,5000000)
plt.xlabel("Steps")
plt.ylabel("Elo")
plt.legend()
plt.savefig(SAVE_DIR / '3-Zoomed-in.pdf')
plt.savefig(SAVE_DIR / '3-Zoomed-in.png')

fig = plt.figure(3)
plt.plot(steps1, smoothed1, label='PPO')
plt.plot(steps2, smoothed2, label='MA-POCA')
plt.plot(steps3, smoothed3, label='A2C')
plt.plot(steps5, smoothed5, label='MPO', color="purple", linewidth=2.0)
plt.ylim(min(min(smoothed1), min(smoothed2), min(smoothed3), min(smoothed5)) - 10, 1760)
plt.xlim(0,50000000)
plt.xlabel("Steps")
plt.ylabel("Elo")
plt.legend()
plt.savefig(SAVE_DIR / '3-Full-view.pdf')
plt.savefig(SAVE_DIR / '3-Full-view.png')

plt.show()