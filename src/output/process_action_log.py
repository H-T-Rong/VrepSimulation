import matplotlib as plt
import pandas as pd

import numpy as np

ACTION_LOG_DIR = './output/evaluation/csv/action_log.csv'
action_logs = pd.read_csv(ACTION_LOG_DIR, index_col=0)

algorithms = ['PID', 'RL']
episode_ceiling = 30

for algorithm_idx, algorithm in enumerate(algorithms):
    for episode_idx in range(episode_ceiling):
        action_log = action_logs.loc[action_logs['algorithm_idx'] == algorithm_idx]
        action_log = action_log[action_logs['episode_idx'] == episode_idx]
        
        
        pass

