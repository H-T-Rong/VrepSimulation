import matplotlib.pyplot as plt
from pandas import DataFrame as df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# 绘制评价图

plt.style.use('_mpl-gallery')

# make data:
# np.random.seed(10)
# D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))




arr = np.array

data = pd.read_csv('./output/evaluation/log.csv', index_col=0)

# # 月平均工作时长概率密度函数估计
# plt.cla()
# fig = plt.figure(figsize=(10,6))
# ax=sns.kdeplot(data['PID_tunerPPO_high_torque_mode:PID'] , color='b',shade=True, label='PID')
# ax=sns.kdeplot(data['PID_tunerPPO_high_torque_mode:RL'] , color='r',shade=True, label='PPO-PID')
# ax.set(xlabel='performance', ylabel='frequency')
# plt.title('performance in electric control - PID V.S. PPO-PID')
# plt.legend()

# plt.savefig('./output/high_torque_mode_performance.jpg')

# plt.cla()
# fig = plt.figure(figsize=(10,6))
# ax=sns.kdeplot(data['PID_tunerPPO_low_torque_mode:PID'] , color='b',shade=True, label='PID')
# ax=sns.kdeplot(data['PID_tunerPPO_low_torque_mode:RL'] , color='r',shade=True, label='PPO-PID')
# ax.set(xlabel='performance', ylabel='frequency')
# plt.title('performance in electric control - PID V.S. PPO-PID')
# plt.legend()

# plt.savefig('./output/low_torque_mode_performance.jpg')

# plt.cla()

cols = list(data.columns)

# high_pid = arr(data['PID_tunerPPO_high_torque_mode:PID'])
# high_rl = arr(data['PID_tunerPPO_high_torque_mode:RL'])

# low_pid = arr(data['PID_tunerPPO_high_torque_mode:PID'])
# low_rl = arr(data['PID_tunerPPO_high_torque_mode:RL'])

for col in cols:
    algorithm_name = col.split(':')[1]
    plot_data = arr(data[col])
    plt.figure(figsize=(10,4))
    mean_episode_reward = np.mean(plot_data)
    
    plt.plot(range(plot_data.shape[0]), [mean_episode_reward] * plot_data.shape[0], label='mean', color='r')
    plt.scatter(range(plot_data.shape[0]), plot_data, label='episode reward')
    plt.xlim((0 - 10, plot_data.shape[0]+10))
    plt.ylim((min(plot_data)-1,max(plot_data)+1))
    plt.xlabel('index')
    plt.ylabel('return')

    plt.legend()

    plt.savefig('./output/evaluation/' + col.replace(':', '') + '/' + algorithm_name + '_scatter.jpg', bbox_inches = 'tight')
    # plt.show()
    plt.cla()
# plot
# D = arr(data)
# fig, ax = plt.subplots()
# VP = ax.boxplot(D, positions=[2, 4, 6, 8], widths=1.5, patch_artist=True,
#                 showmeans=False, showfliers=False,
#                 medianprops={"color": "white", "linewidth": 0.5},
#                 boxprops={"facecolor": "C0", "edgecolor": "white",
#                           "linewidth": 0.5},
#                 whiskerprops={"color": "C0", "linewidth": 1.5},
#                 capprops={"color": "C0", "linewidth": 1.5})

# ax.set(xlim=(0, 6), xticks=np.arange(1, 8),
#        ylim=(-5, 3), yticks=np.arange(1, 8))

# plt.show()


# pass