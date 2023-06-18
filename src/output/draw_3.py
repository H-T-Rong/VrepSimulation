import matplotlib.pyplot as plt
from pandas import DataFrame as df
import pandas as pd
import numpy as np
import os
# 绘制tensorboard输出数据
FILE_PREFIX = 'run-.-tag-rollout_'

# path_1 = './output/stir_log/complete_log/high_torque_env'
# path_2 = './output/stir_log/complete_log/low_torque_env'
# paths = [path_1, path_2]
FILE_PATH = './training/tensorboard_log/csv/PPO_7/'
PIC_OUTPUT_PATH = './training/tensorboard_log/pic/PPO_7/'

paths = [FILE_PATH]

if not os.path.exists(PIC_OUTPUT_PATH):
    os.mkdir(PIC_OUTPUT_PATH)

def smooth_curve(curve_data):
    # curve_data是个有序列表,输出用移动平均法平滑后的曲线数据
    if type(curve_data) == list:
        curve_data = np.array(curve_data)
    smooth_window = 10
    smooth_data = np.zeros(curve_data.shape)
    
    # 滑动窗口大小为5
    for i in range(len(curve_data)):
        try:
            window_data = curve_data[i-int(smooth_window/2):i+int(smooth_window/2):1]
            smooth_data[i] = sum(window_data)/len(window_data)
        except:
            smooth_data[i] = curve_data[i]
    return smooth_data


for path in paths:
    filenames = os.listdir(path=path)
    csv_filenames = []
    for filename in filenames:
        if filename[-3:] == 'csv':
            csv_filenames.append(filename)
    for csv in csv_filenames:

        np.random.seed(1)
        data = pd.read_csv(path + csv)

        x = data['Step']
        y = data['Value']

        std = np.std(np.array(y))
        
        y1 = y + 1.5 * std
        y2 = y - 1.5 * std
        y = smooth_curve(y)
        

        # y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
        # y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))

        # plot
        fig, ax = plt.subplots()

        ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
        ax.plot(x, y, linewidth=2, label=csv[len(FILE_PREFIX):-4].replace('_',' '), color='r')
        # ax.scatter(x[::1], y[::1], s=100, c='r', marker='*',alpha=0.65)
        ax.set(xlim=(np.min(x),np.max(x)),ylim=(np.min(y)-2*std, np.max(y)+2*std))
        ax.legend()
        ax.set_title(csv[len(FILE_PREFIX):-4].replace('_',' '))
        ax.grid()

        # ax.set(xlim=(0, np.max(x)), xticks=np.arange(1, 8),
        #     ylim=(0, np.max(y)), yticks=np.arange(1, 8))
        
        # plt.show()
        plt.savefig(PIC_OUTPUT_PATH+csv[len(FILE_PREFIX):].replace('_',' ')+'.jpg')
