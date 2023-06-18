import socket
import numpy as np
from robot import Robot
import threading
import time

from queue import Queue
from utils import TimeRecoder

from threading import Event

PID_HOST = '127.0.0.1'
PID_PORT = 50008

ENV_HOST = '127.0.0.1'
ENV_PORT = 50009

# 在仿真中PIDController暂时先不需要实现自己的network
class PIDController:
    def __init__(self) -> None:

        self.robot = Robot()
        
        # socket communication
        self.HOST = '127.0.0.0'
        self.PORT = 50008
        
        self.communication_protocol = 'UDP' # TCP

        if self.communication_protocol == 'TCP':
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            # AF_INET：使用IPv4地址簇
            # SOCK_STREAM：使用TCP协议，会出现粘包问题，但可靠传输,
        elif self.communication_protocol == 'UDP':
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise ValueError('No such communication protocol: ', self.communication_protocol)

        # AF_INET：使用IPv4地址簇
        # SOCK_STREAM：使用UDP协议，不会出现粘包问题，但不可靠传输,


        self.sock.bind((PID_HOST, PID_PORT))
        if self.communication_protocol == 'TCP':
            self.sock.listen(1) #在不accept的情况下维护一个长度为1的请求队列
        elif self.communication_protocol == 'UDP':
            self.sock.settimeout(20)  
            #设置一个时间提示，如果10秒钟没接到数据进行提示
        else:
            raise ValueError('No such communication protocol: ', self.communication_protocol)
        
        # PID algorithm
        self.kp = 0.1
        self.kd = 0.01
        self.ki = 0.01

        self.P = 0.0
        self.D = 0.0
        self.I = 0.0

        self.delta_kp = 0.0
        self.delta_kd = 0.0
        self.delta_ki = 0.0

        self.error = None

        # 记录按照目标速度走的话理论上走过的距离（估算）
        self.target_distance = 0.0
        # 记录按照当前速度走的话理论上走过的距离（估算）
        self.actual_distance = 0.0
        # 记录两者之差
        self.error_distance = 0.0


        self.current_velocity = 0.0
        self.target_velocity = 0.0
        self.current_force = 0.0
        self.rotate_link = self.robot.getObjectHandle('rotate_link_respondable')
        self.output_velocity = 0.0

        self.client_conn = None

        # 当为True时，PID算法输出的速度超出阈值，电机即将跑飞
        self.target_velocity_warnings = False

        # 当前目标速度的编号
        self.target_velocity_idx = 0

        # 发送给env数据包的编号
        self.send_package_idx = 0

        # 记录当前目标速度设置后得到的RL进程输出的个数
        self.RL_output_num = 0
        
        # 分别统计三个线程的频率
        self.timeRecorder_1 = TimeRecoder()
        self.timeRecorder_2 = TimeRecoder()
        self.timeRecorder_3 = TimeRecoder()
        self.delay_sec_1 = None
        self.delay_sec_2 = None
        self.delay_sec_3 = None

        self.global_end_flag = False # 全局终止标志位
        self.reset_complete = True

        self.target_velocity_history = []
        # 当为True时，根据random设置目标速度，当为False时，根据已经保存的target_velocity_history设置目标速度
        self.shuffle_target_velocity = True
        self.reset_confirm = False

        self.robot.setVelocity(targetVelocity=0)

        self.permission_for_running = Event() # 条件变量，用于实现线程同步
        self.NoneType_Object = Event() # 条件变量，用于实现线程同步

    def PIDoutput(self, control_error):

        self.D = control_error - self.P
        self.P = control_error
        self.I += control_error

        output = (self.kp + self.delta_kp) * self.P + (self.kd + self.delta_kd) * self.D +(self.ki + self.delta_ki) * self.I 

        return output

    def PIDThread(self):
        """Only interface opened to the client

        Args:
            target_velocity (_type_): _description_
        """
        # set up the time recorder
        self.timeRecorder_1.set()

        # 如果下面启动TCP数据流通信，则这里一定要先request
        if not self.robot.request_dataStreaming(request_types=['Force', 'Velocity'], targetObj=[self.robot.motor, self.rotate_link]):
            raise RuntimeError('request for data Streaming is unsuccessfull')
    
        while(True):
            self.permission_for_running.wait()

            # get random-generated target velocity:
            target_velocity = self.target_velocity

            if abs(self.delta_kp) > 0.001:
                pass
            else:
                self.delta_kp = 0.0
            
            try:
                linear, angular = self.robot.getVelocity(self.rotate_link, opmode='streaming')
            except TypeError:
                print('pidController::PIDThread: Typeerror')
                break

            self.current_force = self.robot.getForce(self.robot.motor, opmode='streaming')
            if angular is None or self.current_force is None:
                self.NoneType_Object.clear()
                continue
            else:
                self.NoneType_Object.set()

            self.current_velocity = angular[2] # 弧度制角速度

            self.error = target_velocity - self.current_velocity

            self.output_velocity += self.PIDoutput(self.error)

            self.target_velocity_warnings = self.robot.setVelocity(targetVelocity=self.output_velocity)

            # print('current velocity: ', self.current_velocity)
            # print('target velocity: ', target_velocity)
            # print("PID output DELAY: ", self.timeRecorder.reset()) # 输出上一次调用reset或者set到目前为止的秒数。

            self.target_distance += target_velocity
            self.actual_distance += self.current_velocity
            self.error_distance = abs(self.target_distance - self.actual_distance)

            if self.global_end_flag:
                break
            
            self.delay_sec_1 = self.timeRecorder_1.reset()
            

    def serverRecvThread(self):

        

        print('PID server recv thread start working ')

        if self.communication_protocol == 'TCP':
            self.client_conn, self.client_addr = self.sock.accept()
            self.client_conn.setblocking(True) # 阻塞模式
            print('Connected by: ', self.client_addr)
        else:
            pass
        self.timeRecorder_2.set()
        while(True):

            if not self.reset_complete:
                continue
            if self.communication_protocol == 'TCP':
                self.data = self.client_conn.recv(1024) #数据缓冲为1024
            elif self.communication_protocol == 'UDP':
                self.data = self.sock.recv(1024)
                
            self.data = self.data.decode('utf-8')

            # 根据顶层通讯协议解析数据：
            recv_buf = self.data.split(':')
            if recv_buf[0] == 'reset':

                self.permission_for_running.clear()
                self.reset_complete = False

                print('pidController: episode done, resetting target velocity')
                self.resetTargetVelocity()
                time.sleep(0.01)
                self.reset_confirm = True # 告诉env，reset已经完成


                self.reset_complete = True
                self.permission_for_running.set()

            elif recv_buf[0] == 'normal':
                str_buf = recv_buf[1].split(',')
                RL_output_idx = eval(str_buf[0])
                if RL_output_idx == self.target_velocity_idx:
                    self.RL_output_num += 1
                    # print('RL meets the requirement of frequency! ')
                    # print('RL output num: ', self.RL_output_num)
                self.delta_kp = eval(str_buf[1])
                self.delta_ki = eval(str_buf[2])
                self.delta_kd = eval(str_buf[3])

            elif recv_buf[0] == 'pause':

                self.robot.pauseSimulation()
            
            elif recv_buf[0] == 'resume':

                self.robot.resumeSimulation()

            elif recv_buf[0] == 'end':
                self.robot.stopSimulation()
                self.robot.exitSimulation()
                print('exit simulation')
                time.sleep(1)
                self.global_end_flag = True
                break

            elif recv_buf[0] == 'evaluation_reset':
                # 以完全相同的目标速度命令序列测试PID和PPO-PID，故这里保存之前的速度序列，然后重启仿真
                self.permission_for_running.clear()

                self.reset_complete = False

                self.robot.setVelocity(targetVelocity=0.0)
                self.robot.stopSimulation()
                
                self.shuffle_target_velocity = False
                print('robot reset')
                self.P = 0.0
                self.D = 0.0
                self.I = 0.0

                self.delta_kp = 0.0

                self.error = None

                # 记录按照目标速度走的话理论上走过的距离（估算）
                self.target_distance = 0.0
                # 记录按照当前速度走的话理论上走过的距离（估算）
                self.actual_distance = 0.0
                # 记录两者之差
                self.error_distance = 0.0


                # self.current_velocity = 0.0
                self.target_velocity = 0.0
                self.current_force = 0.0
                self.output_velocity = 0.0

                time.sleep(0.6)
                self.robot.resumeSimulation()
                
                self.target_velocity_idx = 0

                self.reset_complete = True

                # self.permission_for_running.set() # 等待env进一步调用reset方法
            else:
                raise RuntimeError('pidController: The format of received message is incorrect! ')
            
            self.delay_sec_2 = self.timeRecorder_2.reset()

            pass
    def serverSendThread(self, sleep_time):
        """给rl进程发送数据的线程

        Args:
            sleep_time (_type_): 线程延迟时间
        """
        print('PID server send thread start waiting ')
        if self.communication_protocol == 'TCP':
            while self.client_conn is None:
                pass
        elif self.communication_protocol == 'UDP':
            while self.error is None:
                pass

        print('get client connection, PID server send thread start sending information')
        self.timeRecorder_3.set()
        
        while(True):
            self.permission_for_running.wait() # 等待reset完成

            warn = 0
            if self.target_velocity_warnings:
                warn = 1
                self.target_velocity_warnings = False
            
            if not self.reset_complete:
                continue
            # 发送数据包：'当前速度的编号,当前速度,当前力,当前误差,当前警告,当前积分项,当前微分项'
            try:
                self.NoneType_Object.wait() # 等待CoppeliaSim数据传输到位再开始发送给env
                if self.reset_confirm:
                    send_message = 'reset_confirm:{},{:.4f},{:.4f},{:.4f},{},{:.4f},{:.4f};{}'.format(self.target_velocity_idx,
                                                                        self.current_velocity,
                                                                        self.current_force,
                                                                        self.error,
                                                                        warn,
                                                                        self.I,
                                                                        self.D,
                                                                        self.send_package_idx)
                    self.reset_confirm = False
                else:
                    send_message = 'normal:{},{:.4f},{:.4f},{:.4f},{},{:.4f},{:.4f};{}'.format(self.target_velocity_idx,
                                                                        self.current_velocity,
                                                                        self.current_force,
                                                                        self.error,
                                                                        warn,
                                                                        self.I,
                                                                        self.D,
                                                                        self.send_package_idx)
                self.send_package_idx += 1
            except TypeError:
                raise TypeError('pidController.py::serverSendThread: NoneType Object')
                break
            
            if self.communication_protocol == 'TCP':
                self.client_conn.sendall(send_message.encode('utf-8')) # current_velocity, force
            elif self.communication_protocol == 'UDP':
                self.sock.sendto(send_message.encode('utf-8'), (ENV_HOST, ENV_PORT))


            time.sleep(sleep_time)

            if self.global_end_flag:
                print('Daemon Thread Exiting')
                break
            
            self.delay_sec_3 = self.timeRecorder_3.reset()

            # error_code = os.system('{}'.format(data))
            # if error_code == 0:
            #     print('The command is successfully implemented')
            #     conn.sendall('Done'.encode('utf-8'))
            #     # break
            # else:
            #     conn.sendall('OS.system error code: {}'.format(error_code).encode('utf-8'))
                # break
    
    # def targetVelocityThread(self, delay):
    #     """我开这个线程每0.5秒重新设置一次速度然后训练是不是有那种大病？

    #     Args:
    #         delay (_type_): _description_
    #     """
    #     while(True):
    #         if self.shuffle_target_velocity:
    #             self.target_velocity = np.random.randint(low=30, high=150) * np.pi / 180
    #         else:
    #             self.target_velocity = 90 * np.pi / 180

    #         self.target_velocity_idx += 1
    #         self.RL_output_num = 0
    #         if self.target_velocity_idx > 10000:
    #             self.target_velocity_idx = 0
    #         time.sleep(delay)
    # 是的

    def resetTargetVelocity(self):

        if self.shuffle_target_velocity:
            self.target_velocity = np.random.randint(low=30, high=150) * np.pi / 180
            print('pidController: setting target velocity:  ')
            print('pidController: ', self.target_velocity)
            # 速度的设置范围： 0.5235 到2.6179
        else:
            # self.target_velocity = 90 * np.pi / 180
            self.target_velocity = self.target_velocity_history[self.target_velocity_idx]
            print('pidController: setting target velocity from history: ')
            print('pidController: ', self.target_velocity)
        
        self.target_velocity_history.append(self.target_velocity)

        self.target_velocity_idx += 1
        self.RL_output_num = 0 # 设置了目标速度以后看看在下次设置目标速度设置之前强化学习算法可以输出几次

        if self.target_velocity_idx > 50000: 
            self.target_velocity_idx = 0 #重新对设置的目标速度编号

        self.error = self.target_velocity - self.current_velocity
        self.send_package_idx = 0

        self.reset_complete = True



    def print_frequency_thread(self):
        while(True):
            time.sleep(5)
            if self.delay_sec_1 is not None:
                print('pid thread frequency: ', 1/self.delay_sec_1)
            if self.delay_sec_2 is not None:
                print('recv thread frequency: ', 1/self.delay_sec_2)
            if self.delay_sec_3 is not None:
                print('send thread frequency: ', 1/self.delay_sec_3)
            if self.global_end_flag:
                break
        



if __name__ == "__main__":
    


    # TARGET_THREAD_DELAY = 0.5 # 下一个目标速度产生延迟
    SEND_THREAD_DELAY = 0.02 # 发送线程延迟

    controller = PIDController()
    

    serverThd = threading.Thread(target=controller.PIDThread, args=())# 为数据更新函数单独创建一个线程，与图像绘制的线程并发执行
    serverThd.daemon = True # Daemon 主进程结束则该线程也强制结束
    serverThd.start()  # 线程执行，启用线程的原因见下一行注释

    PIDRecvThd = threading.Thread(target=controller.serverRecvThread, args=())# 创建发送线程
    PIDRecvThd.daemon = True
    PIDRecvThd.start()

    PIDSendThd = threading.Thread(target=controller.serverSendThread, args=(SEND_THREAD_DELAY,))# 创建接受线程
    PIDSendThd.daemon = True
    PIDSendThd.start()

    printThd = threading.Thread(target=controller.print_frequency_thread, args=())
    printThd.daemon = True
    printThd.start()

    # TgtVelThd = threading.Thread(target=controller.targetVelocityThread, args=(TARGET_THREAD_DELAY,))
    # TgtVelThd.daemon = True
    # TgtVelThd.start()

    # test_reward = False
    # if test_reward:
    #     # reward calculation relative
    #     errorQueue_maxSize = 10
    #     error = 0
    #     errors_queue = Queue(maxsize=errorQueue_maxSize)
    #     for i in range(errorQueue_maxSize): # initialize the queue
    #         errors_queue.put(0)

    thdList = [serverThd, PIDRecvThd, PIDSendThd, printThd]

    # 主进程挂起等待线程结束，如果调试时强制终止主进程，则自线程由于是
    # 守护线程(daemon)，故也同时被终止
    # setdaemon()函数和join()联合使用达成这个效果
    # 保证主线程既不在运行时占用其它线程的时间片，也保证调试时强制终止主线程也同时
    # 杀死其它子线程
    for thd in thdList:
        thd.join() 
    


    # while not controller.global_end_flag:
        # pid control testing:
        # controller.permission_for_running.set()
        # controller.target_velocity = 180 * np.pi / 180

        # reward testing:
        # pass






# import socket
# import os

# HOST = '127.0.0.1'
# PORT = 50008
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # 定义socket类型，网络通信，TCP
# s.bind((HOST, PORT))
# s.listen(backlog=1)

# while True:
#     conn, addr = s.accept()
#     print('Connected by: ', addr)
#     while True:
#         data = conn.recv(1024)
#         data = data.decode('utf-8')
#         error_code = os.system('{}'.format(data))
#         if error_code == 0:
#             print('The command is successfully implemented')
#             conn.sendall('Done'.encode('utf-8'))
#             # break
#         else:
#             conn.sendall('OS.system error code: {}'.format(error_code).encode('utf-8'))
#             # break
#     print('connection closing')
#     conn.close()