import gymnasium as gym
import numpy as np
from gymnasium import spaces
from robot import Robot
from queue import Queue
import socket
import threading
import time
from threading import Event

from utils import TimeRecoder

from utils import Normalization
from utils import RewardScaling

from analyst import Analyst
from callback import CustomCallback
from traceback import print_stack
from threading import Lock


N_FEATURES = 6
# 假设观测变量个
# 
N_ACTION = 3
# 假设强化学习算法对P，I，D三个变量都能进行调节

PID_HOST = '127.0.0.1'
PID_PORT = 50008

ENV_HOST = '127.0.0.1'
ENV_PORT = 50009

# 如果目标速度和接收到的当前速度连续40次相对误差在百分之二以内，判断episode终止
DONE_DETERMINE = 10
DONE_RANGE = 5/100
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, logger=None):
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(N_ACTION, ),dtype=np.float32)

        self.observation_space = spaces.Box(low=-100, high=100, shape=(N_FEATURES, ), dtype=np.float32)

        # 动作空间是P值，是个连续的变量，这里设其上限是1，下限是-1,因为要作标准化
        # 然而在step函数里我会给它rescale到一个更小的数值，因为delta_kp的上下限
        # 这么大的时候实测可知电机直接跑飞
        self.kp_rescale = 0.1
        self.ki_rescale = 0.01
        self.kd_rescale = 0.01 # 注意这里要和pidController的self.kp, self.ki, self.kd一致

        self.normalize = Normalization(shape=(N_FEATURES)) # 对observation作标准化
        self.scale_reward = RewardScaling(shape=1, gamma=1) # 对reward作scaling

        self.last_velocity = 0
        self.current_velocity = 0
        self.Force = 0
        self.target_velocity_idx = 0

        self.packageQueue_maxSize = 10
        self.packages_queue = Queue(maxsize=self.packageQueue_maxSize)

        # reward calculation relative
        self.errorQueue_maxSize = 10
        self.error = 0
        self.errors_queue = Queue(maxsize=self.errorQueue_maxSize)
        
        self.velQueue_maxSize = 10
        self.velQueue = Queue(maxsize=self.velQueue_maxSize)

        # for i in range(self.errorQueue_maxSize): # initialize the queue
        #     self.errors_queue.put(0)
        
        self.sleepTime = 0.1 # 根据pidController的sendThd_sleepTime决定
        

        self.communication_protocol = 'UDP' # TCP

        if self.communication_protocol == 'TCP':
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            # AF_INET：使用IPv4地址簇
            # SOCK_STREAM：使用TCP协议，会出现粘包问题，但可靠传输,
        elif self.communication_protocol == 'UDP':
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise ValueError('No such communication protocol: ', self.communication_protocol)


        self.PID_address = (PID_HOST, PID_PORT)
        self.address = (ENV_HOST, ENV_PORT)
        # AF_INET：使用IPv4地址簇
        # SOCK_STREAM：使用UDP协议，不会出现粘包问题，但不可靠传输,


        self.sock.bind(self.address)
        
        if self.communication_protocol == 'TCP':
            self.sock.connect((PID_HOST, PID_HOST)) 

        elif self.communication_protocol == 'UDP':
            self.sock.settimeout(30)
            #设置一个时间提示，如果10秒钟没接到数据进行提示
        else:
            raise ValueError('No such communication protocol: ', self.communication_protocol)
        
        self.step_count = 0
        self.ban_scale_reward = False

        self.pid_Controller_reset_complete = False
        self.reset_complete = Event() # 对reset()和其它操作规定同步
        
        self.step_lock = Lock() # 阻止多线程进入step()引起混乱
        self.reset_lock = Lock() # 阻止多线程进入reset()引起混乱

        self.logger = logger
        self.last_observation = None
        self.last_action = None

        self.done_count = 0 # 满足终止条件的计数
        self.origin_error = None
        self.origin_observation = None

        self.target_velocity = None
        self.timeRecorder = TimeRecoder() # 用于计算上升时间和持续时间

        self.episode_ts = [] #记录每个episode的调节时间
        self.episode_tr = [] #记录每个episode的上升时间
        self.episode_sigma = [] #记录每个episode的超调量
        self.episode_get_tr = False
        self.episode_max_absError = 0 #记录每个episode最大的误差的绝对值

        self.dead_loop = 0 #记录死循环次数

        updateThd = threading.Thread(target=self.socket_communication, args=())
        updateThd.daemon = True
        updateThd.start()

        self.timeRecorder_2 = TimeRecoder()
        self.update_delay = []

        self.analyst = Analyst()


    def step(self, action):

        self.step_lock.acquire()
        self.reset_complete.wait()

        # counting steps in a episode
        self.step_count += 1

        if self.packages_queue.empty():
            while(self.packages_queue.empty()):
                # print('packages_queue empty, env step method wait for packages')
                time.sleep(0.012) # wait for packages update

        # get socket package
        package = self.packages_queue.get()

        # unpack socket package
        message = package[0] # get socket package information
        observation_pacakge = package[1:] # get current observation
        self.target_velocity = observation_pacakge[0]
        self.current_velocity = observation_pacakge[1]
        self.force = observation_pacakge[2]
        self.error = observation_pacakge[3]
        self.velocity_error = self.error
        warn = observation_pacakge[4]
        self.observed_I = observation_pacakge[5]
        self.observed_D = observation_pacakge[6]
        
        # get observation:
        observation = np.array([self.step_count,
                                self.current_velocity,
                                self.velocity_error,
                                self.force,
                                self.observed_I,
                                self.observed_D], dtype=np.float32)
        
        # normalize observation:
        observation = self.normalize(observation)

        # calculate rewards:
        recent_errors_copy = np.array(list(self.errors_queue.queue)).copy()
        recent_errors_copy = np.absolute(recent_errors_copy)
        # reward = - np.mean(recent_errors_copy) * 10 - self.step_count / 0.2 #- abs_distance_error/10 
        reward = -self.step_count

        # rescale reward :
        if not self.ban_scale_reward:
            reward = self.scale_reward(reward)
        else:
            pass

        # 判断episode是否终止，共三种终止条件，前两个为异常终止
        done = False
        if warn:
            # episode end, big failure!
            done = True
            # reward = -1000/self.step_count
            reward = -1000
            self.step_count = 0
            print('done! because of failure to coverage')
        if abs(self.error) < abs(self.origin_error) * DONE_RANGE:
            # 必须保证误差连续不超过 "DONE_DETERMINE" 次，才能认为达到稳定
            self.done_count += 1
            if self.done_count > DONE_DETERMINE:
                # reward = reward + self.done_count / 4
                self.done_count = 0
                done = True
                episode_ts = self.timeRecorder.mark()
                self.episode_ts.append(episode_ts) # calculate ts
                self.episode_sigma.append(self.episode_max_absError / abs(self.origin_error)) # calculate sigma
                if len(self.episode_tr) < len(self.episode_sigma):
                    self.episode_tr.append(0)
                self.pid_Controller_reset_complete = False
                self.reset_complete.clear()
            else:
                done = False
        else:
            self.done_count = 0
        if self.step_count > 500:
            done = True
            print('done because of reaching the end of episodes')
            self.step_count = 0

            # episode_ts = self.timeRecorder.mark()
            # self.episode_ts.append(episode_ts)
            # self.episode_sigma.append(self.episode_max_absError / (abs(self.origin_error) + 1e-3))
            
            self.pid_Controller_reset_complete = False
            self.reset_complete.clear()
        # # calculate tr:
        # if self.episode_get_tr:
        #     pass
        # else:
        #     if self.error * self.origin_error < 0:
        #         episode_tr = self.timeRecorder.mark()
        #         self.episode_tr.append(episode_tr)
        #         self.episode_get_tr = True
        # # sigma calculate relative:  
        # if abs(self.error) > self.episode_max_absError:
        #     self.episode_max_absError = abs(self.error)
        

        # implement action towards the environment:
        send_message = "normal:{},{:.3f},{:.3f},{:.3f}".format(self.target_velocity_idx,
                                                            action[0] * self.kp_rescale, 
                                                            action[1] * self.ki_rescale,
                                                            action[2] * self.kd_rescale)
        if self.communication_protocol == 'UDP':
            self.sock.sendto(send_message.encode('utf-8'), self.PID_address)
        else:
            self.sock.sendall(send_message.encode('utf-8'))

        # logging information into csv, using custom logger
        self.last_velocity = self.current_velocity
        self.pid_Controller_reset_complete = False
        self.last_observation = observation
        self.last_action = action
        if self.logger is not None:
            if self.last_observation is not None:
            # log : (s_t, a_t, s_t+1, r_t+1)
                self.logger.log(self.last_observation.tolist()+\
                                self.last_action.tolist()+\
                                observation.tolist()+\
                                [reward])

        info = {'info':'environment step!'}
        truncated = False

        time.sleep(0.1) 
        # 给step method进行延时，对频率进行控制。
        # 不然这里step会和pidController的控制信号输出频率同频

        self.step_lock.release()

        return observation, reward, done,truncated, info

    def reset(self):
        """初始化环境对象env
        Returns:
            _type_: 返回智能体的初始观测，是np.array对象

        """
        # print_stack() # 打印函数调用栈(调试用)
        
        self.reset_lock.acquire()

        info = {'info':'environment reset'}
        send_message = "reset:666,666,666,666"
        if self.communication_protocol == 'UDP':
            self.sock.sendto(send_message.encode('utf-8'), self.PID_address)
        else:
            self.sock.sendall(send_message.encode('utf-8'))
        print('env.py: reset(): env send reset command')
        
        

        observation = np.array([0.0] * N_FEATURES, dtype=np.float32)
        self.last_velocity = 0
        self.current_velocity = 0
        self.Force = 0
        self.step_count = 0
        self.last_observation = None
        self.last_action = None
        self.done_count = 0
        self.origin_error = None
        self.target_velocity = None
        self.episode_get_tr = False
        self.origin_observation = None

        if self.logger is not None:
            self.logger.write()

        while not self.errors_queue.empty():
            self.errors_queue.get() # 将error队列中的数据清空
        while not self.packages_queue.empty():
            self.packages_queue.get()
        
        # for idx in range(self.errorQueue_maxSize):
        #     self.errors_queue.get()
        #     self.errors_queue.put(0.0)

        self.scale_reward.reset()

        # while not self.pid_Controller_reset_complete:
        #     time.sleep(1)
        #     if self.communication_protocol == 'UDP':
        #         self.sock.sendto(send_message.encode('utf-8'), self.PID_address)
        #     else:
        #         self.sock.sendall(send_message.encode('utf-8'))

        #     pass

        self.reset_complete.wait() # wait until reset complete
        self.timeRecorder.begin()

        print('env.py: reset(): get reset confirm message, reset complete')

        if self.origin_observation is None:
            raise RuntimeError('env.py:: reset(): reset failure')
        else:
            observation = self.origin_observation

        
        # send_message = "resume:666"
        # if self.communication_protocol == 'UDP':
        #     self.sock.sendto(send_message.encode('utf-8'), self.PID_address)
        # else:
        #     self.sock.sendall(send_message.encode('utf-8'))
        self.reset_lock.release()

        return (observation,info)  # reward, done, info can't be included

    def render(self):
        """以图形化的形式显示环境当前的状态，现在在用vrep，没有
        使用这个函数的必要
        """
        pass

    def close(self):
        send_message = "end:666,666,666,666"
        self.sock.sendto(send_message.encode('utf-8'), self.PID_address)
        pass

    def socket_communication(self):
        """线程的handle函数，开启这个线程的原因在于
        ，step函数的频率要远远慢与PID控制器输出的频率，这
        个线程的初衷是尽可能地跟上pidController的输出频率
        以PID控制器的输出频率更新error数据
        """
        print('env.py: RL process socket communication enabled ! ')

        # 计算update thread的频率
        # self.timeRecorder_2.set()

        while(True):
            if self.communication_protocol == 'UDP':
                # get current observation
                str_buf = self.sock.recv(1024)
                str_buf = str_buf.decode('utf-8')

                # print('env received messages: ', str_buf)
                str_buf = str_buf.split(':')
                message = str_buf[0]
                # observations = str_buf[1]
                str_buf2 = str_buf[1].split(';')
                observations = str_buf2[0]
                package_idx = eval(str_buf2[1])
                
                buf = observations.split(',')

                current_velocity = eval(buf[1])
                error = eval(buf[3])

                package = (message, 
                           eval(buf[0]),
                           eval(buf[1]),
                           eval(buf[2]),
                           eval(buf[3]),
                           eval(buf[4]),
                           eval(buf[5]),
                           eval(buf[6]))
                
                if not self.packages_queue.full():
                    self.packages_queue.put(package)
                else:
                    self.packages_queue.get()
                    self.packages_queue.put(package)

                if not self.errors_queue.full():
                    self.errors_queue.put(error)
                else:   
                    self.errors_queue.get()
                    self.errors_queue.put(error)
                
                # 当再次接收到pidController发来的error数据时，认为reset完毕
                if not self.pid_Controller_reset_complete:
                    if message == 'reset_confirm':
                        if package_idx == 0:
                            pass
                        else:
                            raise RuntimeError('package idx is not correct, socket communication needs checking')
                        
                        print('env.py: socket_communication(): pidController reset complete, judging from reset_confirm')
                        origin_error = error
                        self.origin_error = origin_error
                        observations_package = package[1:]
                        self.origin_observation = np.array([0,
                                                            observations_package[1],
                                                            observations_package[3],
                                                            observations_package[2],
                                                            observations_package[5],
                                                            observations_package[6]], dtype=np.float32)

                        if origin_error < 1e-3:
                            origin_error = 1e-3
                        
                        self.current_velocity = current_velocity
                        self.target_velocity = self.origin_error + self.current_velocity

                        self.pid_Controller_reset_complete = True
                        self.reset_complete.set()

                        self.dead_loop = 0
                    else:
                        self.dead_loop += 1
                        if self.dead_loop > 3000:
                            raise RuntimeError('reset environment fail')
                        continue
            # 计算update thread的频率
            # delay = self.timeRecorder_2.reset()
            # self.update_delay.append(delay)
            # if len(self.update_delay) > 10:
            #     print('env: update package frequency:', 1/np.mean(self.update_delay))
            #     self.update_delay.clear()

    def ban_reward_rescale(self, bol):
        if type(bol) == bool:
            self.ban_scale_reward = bol
        else:
            raise TypeError('the type of bol is incorrect')
    
    def get_error(self):
        """用于返回标准化前的error数据

        Returns:
            _type_: _description_
        """
        return self.error
    
    def get_original_error(self):

        return self.origin_error
    
    def get_target_velocity(self):

        if self.target_velocity is not None:
            return self.target_velocity
        else:
            return None
    
    def get_episode_evaluation_data(self):
        return (self.episode_tr, self.episode_ts, self.episode_sigma)
    
    def evaluation_reset(self):
        self.target_velocity_idx = 0

        send_message = "evaluation_reset:{},{:.3f},{:.3f},{:.3f}".format(self.target_velocity_idx,
                                                                         0,
                                                                         0,
                                                                         0)
        
        if self.communication_protocol == 'UDP':
            self.sock.sendto(send_message.encode('utf-8'), self.PID_address)
        else:
            self.sock.sendall(send_message.encode('utf-8'))

        self.reset()
        
        self.episode_tr.clear()
        self.episode_sigma.clear()
        self.episode_ts.clear()

        time.sleep(1)

    # def set_obstacle(self):
    #     self.robot.add


    # def modifyPID(self, delta_kp):
    #     self.kp += delta_kp

    # def PIDoutput(self, control_error):
    #     self.D = control_error - self.P
    #     self.P = control_error
    #     self.I += control_error

    #     output = self.kp * self.P + self.ki * self.I + self.kd * self.D

    #     return output
        
    # def PIDprocess(self):
    #     # generate random target velocity:
    #     target_velocity = np.random.randint(low=0, high=180-1)

    #     linear, angular = self.robot.getVelocity('motor')

    #     self.current_velocity = angular[2]

    #     self.error = target_velocity - self.current_velocity

    #     _ = self.errors_queue.get()

    #     self.errors_queue.put(self.error)

    #     output_velocity = self.PIDoutput(self.error)

    #     self.robot.setVelocity(targetVelocity=output_velocity)