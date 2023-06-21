import pandas as pd
import numpy as np
from pidController import PIDController
from utils import TimeRecoder
import matplotlib.pyplot as plt



class Analyst:
    def __init__(self) -> None:
        self.df = pd.DataFrame(columns=['idx', 'error', 'time'])
        self.tr_step = -1 # 记录上升时间步数
        self.ts_step = -1 # 记录调节时间步数
        self.sigma = None # 记录超调量
        self.ts = None # 记录调节时间
        self.tr = None # 记录上升时间
        self.preprocess_complete = False # 在计算sigma，ts，tr前判断是否调用过preprocess函数
        self.sign_consistent = False #判断error序列所有error符号是否一致对计算sigma和tr很重要
        self.error_arr = None
        self.range = 0 # 记录数据条目数

    def append(self, df_data:pd.DataFrame):
        
        if not isinstance(df_data, pd.DataFrame):
            self.df = pd.concat([self.df, pd.DataFrame(df_data, columns=['idx','error'])])
        else:
            self.df = pd.concat([self.df, df_data])
        self.range += 1
    
    def preprocess(self):
        """对数据作预处理，为计算上升时间，调节时间，超调量作准备

        Raises:
            RuntimeError: _description_
        """
        if not self.df.shape[0] > 1:
            raise RuntimeError('the num of steps of an episode should be more than 1')
        
        self.df.reset_index(drop=True)
        self.error_arr = np.array(self.df['error'])
        self.df = self.df.sort_values(by='idx', ascending=True)
        origin_error = self.error_arr[0]
        

        if origin_error > 0:
            _bol_temp_positive = (self.error_arr > 0)
            if _bol_temp_positive.all(): # 符号全部为正
                self.sign_consistent = True
            else:
                self.sign_consistent = False
        else:
            _bol_temp_negative = (self.error_arr < 0)
            if _bol_temp_negative.all(): # 符号全部为负
                self.sign_consistent = True
            else:
                self.sign_consistent = False
        
        self.preprocess_complete = True

    
    def get_ts(self):
        """计算调节时间

        Returns:
            _type_: _description_
        """

        if self.ts is None:
            if self.preprocess_complete:
                pass
            else:
                self.preprocess()
            
            abs_origin_error = abs(self.error_arr[0])

            # 计算持续时间
            for i,error in enumerate(self.error_arr):
                _bol_temp = (np.abs(self.error_arr[i:]) < (abs_origin_error * 5/100))
                if _bol_temp.all():
                    self.ts_step = i
                    break
                else:
                    continue
            self.ts = self.df['time'].iloc[self.ts_step]
        else:
            pass

        return self.ts

    def get_tr(self):
        """计算上升时间

        Returns:
            _type_: 上升时间
        """
        if self.tr is None:
            if self.preprocess_complete:
                pass
            else:
                self.preprocess()
            self.get_ts()
            if not self.sign_consistent:
                # 计算上升时间
                origin_error = self.error_arr[0]
                if origin_error < 0:
                    for i, error in enumerate(self.error_arr):
                        if error > 0:
                            self.tr_step = i
                            break
                        else:
                            continue
                else:
                    for i, error in enumerate(self.error_arr):
                        if error < 0:
                            self.tr_step = i
                            break
                        else:
                            continue
                self.tr = self.df['time'].iloc[self.tr_step]
                if self.tr > self.ts:
                    self.tr_step = self.ts_step
                    self.tr = self.ts
            else:
                
                self.tr_step = self.ts_step
                self.tr = self.ts
        else:
            pass

        return self.tr


    def get_sigma(self):
        """计算超调量

        Returns:
            _type_: 超调量
        """
        if self.sigma is None:
            if self.preprocess_complete:
                pass
            else:
                self.preprocess()
            
            
            if self.sign_consistent:
                self.sigma = 0
            else:
                abs_error_arr = np.abs(self.error_arr[1:])
                abs_max_idx = np.argmax(abs_error_arr)
                self.sigma = abs(self.error_arr[abs_max_idx])/ abs(self.error_arr[0])
        else:
            pass

        return self.sigma


    def reset(self):
        """每过一个episode就要重置一次
        """
        self.df = pd.DataFrame(columns=['idx', 'error', 'time'])
        self.tr_step = -1#记录上升时间
        self.ts_step = -1#记录调节时间
        self.sigma = None #记录超调量
        self.ts = None
        self.tr = None
        self.sort_complete = False
        self.error_arr = None
        self.range = 0

    def illustrate(self, save_path=None, show=False):

        num_episode_steps = self.range # PID算法产生的控制信号的次数

        plt.plot(range(num_episode_steps), np.ones(num_episode_steps)* target_vel, label='target velocity')
        plt.plot(range(num_episode_steps), np.ones(num_episode_steps)* target_vel - analyst.error_arr, label='current velocity')
        plt.xlabel('step')
        plt.ylabel('error')
        plt.grid()
        plt.legend()
        if show:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path)

        pass

if __name__ == "__main__":


    controller = PIDController()
    controller.kp = 0.5
    controller.ki = 0.02
    controller.kd = 0.02

    current_vel = 0
    target_vel = 90 * np.pi / 180
    timerecorder = TimeRecoder()
    analyst = Analyst()

    timerecorder.begin()

    num_episode_steps = 500

    # episode start
    
    for step in range(num_episode_steps):

        # episode duration 

        current_error = target_vel - current_vel

        control_vol = controller.PIDoutput(current_error)

        current_vel += control_vol
        
        time = timerecorder.mark()

        analyst.append(pd.DataFrame([[step, current_error, time]], columns=['idx','error','time']))

    # episode end and calculate

    analyst.preprocess()
    ts = analyst.get_ts()
    tr = analyst.get_tr()
    sigma = analyst.get_sigma()
    
    # episode plot
    analyst.illustrate(show=True)

    analyst.reset()

    pass