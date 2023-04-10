#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:21:52 2020

@author: czh513
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

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
    print(data.shape)
    scalar = data['Value'].values
    last = scalar[0]
    print(type(scalar))
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    # save = pd.DataFrame({'Step':data['Step'].values, 'Value':smoothed})
    # save.to_csv('smooth_' + csv_path)

    steps = data['Step'].values
    steps = steps.tolist()
    origin = scalar.tolist()

    #fig = plt.figure(1)
    # plt.plot(steps, origin, label='origin')
    #plt.plot(steps, smoothed, label='smoothed')
    # plt.ylim(0, 220) # Tensorboard中会滤除过大的数据，可通过设置坐标最值来实现
    #plt.legend()
    #plt.show()
   
   # 为了使线形更加的平滑可以使用聚合功能，表示对x变量的相同值进行多次测量，取平均，并取可信区间
   # fmri = sns.load_dataset("data")
   # fig2=plt.figure(2)
    print(data)
    
  
    endrewards=smoothed[180:200]
    
    return endrewards, steps, smoothed
    #sns.relplot(x="Step", y="Value", kind="line", ci="sd", data=data3)
  
    #sns.set(style="ticks", color_codes=True)
    #sns.catplot(x="Step", y="Value", kind="box", data=data3)
    

do_minimum = True
endrewards1, steps1, smoothed1 =smooth_and_plot('data/ppo-new.csv')
endrewards2, steps2, smoothed2 =smooth_and_plot('data/poca.csv')
endrewards3, steps3, smoothed3 =smooth_and_plot('data/a2c.csv')
endrewards4, steps4, smoothed4 =smooth_and_plot('data/dqn.csv')

print(len(steps1))
print(len(steps2))
print(len(steps3))
print(len(steps4))
print(len(smoothed1))
print(len(smoothed2))
print(len(smoothed3))
print(len(smoothed4))
print(len(endrewards1))
print(len(endrewards2))
print(len(endrewards3))
print(len(endrewards4))

if do_minimum:
    minimum = min(len(steps1), len(steps2), len(steps3))
    steps1 = steps1[:minimum]
    steps2 = steps2[:minimum]
    steps3 = steps3[:minimum]
    steps4 = steps4[:minimum]
    smoothed1 = smoothed1[:minimum]
    smoothed2 = smoothed2[:minimum]
    smoothed3 = smoothed3[:minimum]
    smoothed4 = smoothed4[:minimum]

    if minimum < 200:
        endrewards1 = smoothed1[-20:]
        endrewards2 = smoothed2[-20:]
        endrewards3 = smoothed3[-20:]
        endrewards4 = smoothed4[-20:]


fig = plt.figure(2)
plt.plot(steps1, smoothed1, label='PPO')
plt.plot(steps2, smoothed2, label='MA-POCA')
plt.plot(steps3, smoothed3, label='A2C')
plt.plot(steps4, smoothed4, label='DQN')
plt.ylim(min(min(smoothed1), min(smoothed2), min(smoothed3), min(smoothed4)) -10, 1350)
plt.xlim(0,5000000)
plt.xlabel("Steps")
plt.ylabel("Elo")

fig = plt.figure(3)
plt.plot(steps1, smoothed1, label='PPO')
plt.plot(steps2, smoothed2, label='MA-POCA')
plt.plot(steps3, smoothed3, label='A2C')
plt.plot(steps4, smoothed4, label='DQN')
plt.ylim(min(min(smoothed1), min(smoothed2), min(smoothed3), min(smoothed4)) - 10, 1760)
plt.xlim(0,50000000)
plt.xlabel("Steps")
plt.ylabel("Elo")

#plt.plot(steps3, smoothed3, label='PPO (O extended)')
#plt.plot(steps4, smoothed4, label='RLHC (O extended)')
plt.legend()
plt.show()

# # Python Independent Sample T-Test
#fig = plt.figure(3)
#sns.kdeplot(endrewards1, shade=True)
#sns.kdeplot(endrewards2, shade=True)
#sns.kdeplot(endrewards3, shade=True)
#sns.kdeplot(endrewards4, shade=True)

#plt.title("Independent Sample T-Test")

#tStat, pValue = stats.ttest_ind(endrewards1, endrewards2, equal_var = False) #run independent sample T-Test
#print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) 
#plt.show()