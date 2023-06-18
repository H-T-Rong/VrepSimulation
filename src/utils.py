import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def num_to_str(x):

    return "{}".format(x)


class RunningMeanStd:
    # Dynamically calculate mean and std
    # observation的shape应该是（sample， feature）
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

class TimeRecoder:
    def __init__(self) -> None:
        
        self.set_time = None
        self.reset_time = None
        self.init_time = None
    
    def set(self):
        self.set_time = time.perf_counter()

    def reset(self):
        self.reset_time = time.perf_counter()
        if self.set_time is not None:
            _temp = self.set_time
        else:
            raise RuntimeError("Time Recorder not set yet")
        self.set_time = time.perf_counter()

        return (self.reset_time - _temp)
    
    def begin(self):
        self.init_time = time.perf_counter()
    
    def mark(self):
        return time.perf_counter() - self.init_time
    
        

    
class Artist:
    def __init__(self, data, pic_type, save_path) -> None:
        self.data = dict()
        self.pic_type = ''
        self.supported_pic_types = ['']

    def draw():
        pass
        
    def save(self):
        pass

class CustomLogger:
    """我自己写的这个类可以包在with语句里用于写csv，语句块退出以后可以自动
    调用close。
    同时只在调用write时将内存中的数据写入外存中的文件，保证不在循环关键处消耗
    时间复杂度。
    写入外存后会删除内存中的数据
    我个人觉得非常好用，尤其是在强化学习仿真环境中采集数据时
    """
    def __init__(self, file_path) -> None:
        print('CustomLogger: init custom logger ! ')
        log_name = 'env_data.csv'
        self.data = []
        self.file_handle = open(file_path + log_name, 'a')
        self.n_sample = 0
        
        # return self

    def __enter__(self):
        print("CustomLogger: __enter__() called ! ")
        return self
    
    def log(self, sample:list):
        """记录每一个sample： (input: shape=(9,), output: shape=(7,))

        Args:
            sample (_type_): _description_
        """

        self.data.append(sample)
        pass

    def write(self):
        """ move data from memory to hard disk
        """
        for sample in self.data:
            self.file_handle.write(', '.join(list(map(num_to_str, sample))) + '\n')
        self.data.clear()

        # clear data in memory:

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("CustomLogger: __exit__() called! ")

        self.file_handle.close()
        print('file_close ! ')


def getLogger():
    return CustomLogger()
    


import sklearn
import random
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
import torch

def get_data(temp):
    # map()的handle函数
    return temp[0]

def get_label(temp):
    # map()的handle函数
    return temp[1]

class Dataset:
    """定义数据集类
    """
    def __init__(self, data, label) -> None:
        if type(data) != np.ndarray:
            data = np.array(data)
        if type(label) != np.ndarray:
            label = np.array(label).squeeze()
        
        if label.shape[0] != data.shape[0]:
            raise ValueError('样本数据多少和标签多少不一致')
        self.data = data
        self.label = label

    def __len__(self):

        return self.data.shape[0]
    def __getitem__(self, key):

        return self.data[key], self.label[key]


class DataLoader:
    """定义数据加载器类
    """
    def __init__(self, dataset:list, batch_size:int, shuffle=True, using_pytorch=False) -> None:
        """_summary_

        Args:
            dataset (list): 下标0对应自变量，下标1对应因变量
            batch_size (_type_): _description_
            shuffle (bool, optional): _description_. Defaults to True.
        """
        if shuffle == True:
            # map_handle_1 = map(get_data, sklearn.utils.shuffle(dataset))
            # map_handle_2 = map(get_label, sklearn.utils.shuffle(dataset))

            # data = list(map_handle_1)
            # label = list(map_handle_2)
            data = sklearn.utils.shuffle(dataset[0])
            label = sklearn.utils.shuffle(dataset[1])



            self.shuffle = True
        if type(data) != np.ndarray:
            data = np.array(data, dtype=np.float32)
        if type(label)!= np.ndarray:
            label = np.array(label, dtype=np.float32)

        
        self.dataset = Dataset(data, label)
        self.batch_size = batch_size
        self.current_idx = 0
        self.iter_terminate = False
        self.using_pytorch = using_pytorch

    def __iter__(self):
        """如果想要一个对象成为一个可迭代对象,既可以使用for,那么必须实现__iter__方法"""

        return self
    

    def __next__(self):
        if self.iter_terminate:
            if self.shuffle == True:
                map_handle_1 = map(get_data, sklearn.utils.shuffle(self.dataset))
                map_handle_2 = map(get_label, sklearn.utils.shuffle(self.dataset))

                data = list(map_handle_1)
                label = list(map_handle_2)

                self.dataset = Dataset(data=data, label=label)
                
            self.iter_terminate = False
            raise StopIteration
            return 
        
        current_idx = self.current_idx
        
        if current_idx + self.batch_size < len(self.dataset):
            data_batch, label_batch = self.dataset[current_idx:current_idx+self.batch_size]
            self.current_idx += self.batch_size
        else:
            left_num = len(self.dataset) - self.current_idx

            prev_data, prev_labels = self.dataset[:current_idx]

            data_scrap = random.choices(prev_data,k=left_num)
            label_scrap = random.choices(prev_labels,k=left_num)

            data_batch, label_batch = self.dataset[current_idx:]
            data_batch = np.vstack([data_batch, data_scrap])
            label_batch = np.array(label_batch.tolist()+label_scrap)

            self.current_idx = 0
            self.iter_terminate = True
            
        if self.using_pytorch:
            return torch.tensor(data_batch, dtype=torch.float32, requires_grad=False), torch.tensor(label_batch, dtype=torch.float32, requires_grad=False)
        else:
            return data_batch, label_batch
    def __len__(self):
        return len(self.dataset)
    
def smooth_curve(curve_data):
    """曲线平滑小道具，画图前使用它

    Args:
        curve_data (列表): 有序列表数据

    Returns:
        列表: 平滑后的列表数据
    """
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

def split_data(xdata, ydata):
    """将数据分为训练集和测试集

    Args:
        xdata (pd.DataFrame): 数据
        ydata (pd.DataFrame): 标签

    Returns:
        pd.DataFrame: 划分后的训练集和测试集
    """
    x_y = pd.concat([xdata, ydata], axis=1)
    train, test = train_test_split(x_y, train_size=0.8, shuffle=True)
    
    x_train = train.iloc[:,:train.shape[1]-1]
    y_train = train.iloc[:,train.shape[1]-1]

    x_test = train.iloc[:,:train.shape[1]-1]
    y_test = train.iloc[:,train.shape[1]-1]

    return x_train, y_train, x_test, y_test

def encode_label(ydata):
    """将分类标签进行编码

    Args:
        ydata (unknoen): 是一个由整型变量组成的线性表

    Returns:
        torch.tensor: 经过one-hot编码的标签
    """
    # 对标签进行one-hot编码
    unique_y = np.sort(np.unique(ydata)).tolist()
    if type(ydata) == pd.Series:
        ydata = np.array(ydata).tolist()


    for i, y in enumerate(ydata):
        ydata[i] = unique_y.index(y)
    ydata = one_hot(torch.tensor(ydata, requires_grad=False))
    return ydata

def decode_label(label):
    """对one-hot编码的标签进行解码

    Args:
        label (unknown): 是一个one-hot编码矩阵

    Returns:
        numpy.ndarray: 解码后的标签向量
    """
    if type(label) == pd.Series:
        label = np.array(label)
    if type(label) == torch.tensor:
        label = label.numpy()
    if type(label) != np.array:
        label = np.array(label)
    index = np.argmax(label, axis=1)
    return index

class NormalizeHelper:
    def __init__(self, method='MinMaxScaler') -> None:
        if method == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif method == 'StdScaler':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.data = None


    def fit(self, data):
        """计算当前数据集的标准化并保存标准化用到的平均值和标准差

        Args:
            data (_type_): _description_

        Returns:
            _type_: 返回标准化后的数据
        """
        self.data = data
        return self.scaler.fit_transform(data)
    
    def reverse_normalize(self, data):
        if self.data is None:
            raise ValueError('Please normalize the data first!! ')
        else:
            return self.scaler.inverse_transform(data)

    def normalize(self, data):
        """用已保存的标准差和平均值标准化数据

        Args:
            data (_type_): _description_

        Returns:
            _type_: 返回标准化后的数据
        """
        return self.scaler.transform(data)

    
if __name__ == "__main__":
    # test normalization
    Scaler = NormalizeHelper(method='StdScaler')

    x = np.array([[1,2,3],[4,5,6],[2,4,5]])
    nor_x = Scaler.fit(x)
    print(nor_x)

    print(Scaler.reverse_normalize(nor_x))
    print(Scaler.normalize(data=x[1:,:]))

    # testing file logger:

    # with CustomLogger() as logger:
    #     for i in range(10):
    #         for i in range(10):
    #             logger.log([666,666,666])
    #         print('data len: ', len(logger.data))
    #         logger.write()
            
    #     pass

    # testing DataLoader:
    
