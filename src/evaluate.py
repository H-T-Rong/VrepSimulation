from env import CustomEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from env import CustomEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.core.frame import DataFrame as df
import threading
from utils import TimeRecoder
import time

MODEL_LOAD_PATH = './complete_model/version_6_12/'
OUTPUT_BASE_PATH = './output/evaluation/'
LOG_OUTPUT_PATH= OUTPUT_BASE_PATH + 'csv/'
PIC_OUTPUT_PATH = OUTPUT_BASE_PATH + 'pic/'


def fix_log(log):
    # check log available for dataframe convertion:
    lens = []
    keys = []
    for key,values in log.items():
        lens.append(len(values))
        keys.append(key)
    if min(lens) != max(lens):
        fix_idx = np.argmin(np.array(lens))
        log[keys[fix_idx]].append(np.mean(log[keys[fix_idx]]))
        done = False
    else:
        done = True

    return done

if __name__ == "__main__":

    if not os.path.exists(LOG_OUTPUT_PATH):
        os.makedirs(LOG_OUTPUT_PATH)
    if not os.path.exists(PIC_OUTPUT_PATH):
        os.makedirs(PIC_OUTPUT_PATH)



    reward_log = dict() # 记录PID和RL在强干扰环境和若干扰环境下的获得的reawrd的对比
    action_log = dict() # 记录error，以及agent对于这样的error是如何调节PID的
    evaluation_log = dict() #记录每个episode的各项指标,包括上升时间，调节时间和超调量    
    parent_path = './output/'
    # modelNames = ['PID_tunerPPO_high_torque_mode', 'PID_tunerPPO_low_torque_mode']
    modelNames = ['PID_tunerPPO']
    algorithms = ['PID', 'RL']
    action_log_types = ['error', 'delta_kp', 'delta_ki', 'delta_kd', 'algorithm_idx', 'episode_idx']

    for log_type in action_log_types:
        action_log[log_type] = []



    algorithm_idx = 0
    timeRecoder = TimeRecoder()

    vrepEnv = CustomEnv()
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    vrepEnv.ban_reward_rescale(False)

    # updateThd = threading.Thread(target=vrepEnv.error_queue_update, args=())
    # updateThd.daemon = True
    # updateThd.start()

    for modelName in modelNames:
        for algorithm_idx,algorithm in enumerate(algorithms):

            # output_path = './output/'+modelName+algorithm
            # if os.path.exists(output_path):
            #     pass
            # else:
            #     os.makedirs(output_path)

            model = PPO.load(MODEL_LOAD_PATH + modelName, env=vrepEnv)

            # Evaluate the agent
            # NOTE: If you use wrappers with your environment that modify rewards,
            #       this will be reflected here. To evaluate with original rewards,
            #       wrap environment in a "Monitor" wrapper before other wrappers.

            print('evaluate.py: start evaluating the policy ! ')

            # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
            # print('evaluation complete! ')
            # print('mean reward: ', mean_reward)
            # print('std_reward', std_reward)
            # Enjoy trained agent
            vec_env = model.get_env()
            obs = vec_env.reset()
            

            print('evaluate.py: initial reset complete')
            purePID = True if algorithm == 'PID' else False
            rewards = []
            episode_mean_reward = []
            

            for episode_idx in range(20):
                timeRecoder.set()
                delay_secs=  []
                while(True):
                    if purePID:
                        # # KP = 0.5, KI = 0.5, KD = 0.5
                        action = np.array([0.0, 0.0, 0.0]).reshape([1,3])

                        # KP = 0.1, KI = 0.02, KD = 0.02
                        # action = np.array([-0.8, -0.96, -0.96]).reshape([1,3])
                    else:
                        action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = vec_env.step(action)

                    if done:
                        rewards = np.array(rewards)
                        episode_mean_reward.append(np.mean(rewards))
                        rewards = []
                        print('evaluate.py: episode done !')
                        break
                    else:
                        rewards.append(reward)
                    error = vec_env.env_method('get_error') # 这个方法可以调用自定义环境中的自己写的方法
                    action_log['error'].append(error[0]) # 注意这里记录的error是经过动态标准化处理的error
                    log_action = action.squeeze().tolist()
                    action_log['delta_kp'].append(log_action[0])
                    action_log['delta_ki'].append(log_action[1])
                    action_log['delta_kd'].append(log_action[2])
                    action_log['episode_idx'].append(episode_idx)
                    action_log['algorithm_idx'].append(algorithm_idx)
                    delay_secs.append(timeRecoder.reset())
                
                print('agent average step frequency: ', 1/np.mean(delay_secs))
            
            (episode_tr, episode_ts, episode_sigma) = vec_env.env_method('get_episode_evaluation_data')[0]
            evaluation_log['episode_tr' + '_' + algorithm] = episode_tr.copy()
            evaluation_log['episode_ts' + '_' + algorithm] = episode_ts.copy()
            evaluation_log['episode_sigma' + '_' + algorithm] = episode_sigma.copy()
            print(algorithm + 'algorithm evaluation complete')

            vec_env.env_method('evaluation_reset')
            time.sleep(1)

            episode_num = len(episode_mean_reward)
            mean_episode_mean_reward = np.mean(episode_mean_reward)

            plt.cla()
            plt.plot(range(episode_num), episode_mean_reward, label='episode reward')
            plt.plot(range(episode_num), [mean_episode_mean_reward] * episode_num, label='mean')
            plt.xlabel('episode')
            plt.ylabel('mean reward')
            plt.legend()
            # plt.show()
            plt.savefig(PIC_OUTPUT_PATH + modelName + ':' + algorithm +'{}.jpg'.format('PID' if purePID else 'RL'))

            reward_log[modelName + ':'  + algorithm] = episode_mean_reward


    # closing environment
    vec_env.env_method('close')

    done_log_fix = False
    while not done_log_fix:
        done_log_fix = fix_log(reward_log)
        # 四种情况下记录reward数据的多少可能有一点点不同导致无法放在一张表格
        # ，故对reward数据较少的列进行填充，直到四个列行数都相同
    while not done_log_fix:
        done_log_fix = fix_log(evaluation_log)

    df_reward_log = df(reward_log)
    df_action_log = df(action_log)
    df_evaluation_log = df(evaluation_log)
    pass
    

    df_reward_log.to_csv(LOG_OUTPUT_PATH + '/reward_log.csv')
    df_action_log.to_csv(LOG_OUTPUT_PATH + './action_log.csv')
    df_evaluation_log.to_csv(LOG_OUTPUT_PATH + './evaluation_log.csv')