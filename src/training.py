from env import CustomEnv
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import os
import random

MODEL_SAVE_PATH = './model/'
TENSORBOARD_PATH = './output/training/tensorboard_log/'
LOG_WRITE_PATH = './output/training/env_model_data/'

WITH_MASSIVE_OBSATCLES = True

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# after environment debug, training of the agent:
# avialable algorithm for contiunous action space:
# ARS, A2C, DDPG, HER, PPO, RecurrentPPO, TD3, TQC, TRPO

# PPO implementation:
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG

from utils import CustomLogger

import threading 
# algorithmTypes = ['PPO', 'SAC', 'TD3', 'DDPG']
algorithmTypes = ['PPO']

# ======================================================
# select the type of algorithm !
algorithmType = 'PPO'
# =====================================================
print('start the environment update thread! ')


with CustomLogger(LOG_WRITE_PATH) as logger:
    vrepEnv = CustomEnv(logger=logger)

    # debug enviroment:
    # check_env(vrepEnv)

    # updateThd = threading.Thread(target=vrepEnv.error_queue_update, args=())
    # updateThd.daemon = True
    # updateThd.start()

    for algorithmType in algorithmTypes:
        if algorithmType == 'PPO':
            print("start training the model ! ")

            # Instantiate the agent
            model = PPO("MlpPolicy", vrepEnv, verbose=1, tensorboard_log=TENSORBOARD_PATH,
                        n_steps=32, batch_size=16)
            # n_steps : num of steps per update of network, default is 2048, clearly it
            # is not realistic here
            # We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.

        elif algorithmType == 'SAC':
            model = SAC("MlpPolicy", vrepEnv, verbose=1, tensorboard_log=TENSORBOARD_PATH,
                        batch_size=32)

        elif algorithmType == 'DDPG':
            model = DDPG("MlpPolicy", vrepEnv, verbose=1, tensorboard_log=TENSORBOARD_PATH,
                        batch_size=32)

        elif algorithmType == 'TD3':
            model = TD3("MlpPolicy", vrepEnv, verbose=1, tensorboard_log=TENSORBOARD_PATH,
                        batch_size=32)
        elif algorithmType == 'PID':
            model = PPO("MlpPolicy", vrepEnv, verbose=1, tensorboard_log=TENSORBOARD_PATH,
                        n_steps=32, batch_size=16)
            vrepEnv.kp_rescale = 0.0
            vrepEnv.ki_rescale = 0.0
            vrepEnv.kd_rescale = 0.0

        wrapped_env = model.get_env()
        print('现在算法的env的在内部被封装数据类型为', type(wrapped_env))
        print('vec_env中包含的env个数为:', wrapped_env.num_envs)

        # Train the agent and display a progress bar
        # model.learn(total_timesteps=int(15e4), progress_bar=True)
        model.learn(total_timesteps=int(5e4), progress_bar=True)
        
        # Save the agent
        model.save(MODEL_SAVE_PATH + "PID_tuner" + algorithmType)

        print('model train complete')
    vrepEnv.close()
    pass



