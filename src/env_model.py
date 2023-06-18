from utils import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.optim as optim
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import NormalizeHelper
import pickle

ENV_MODEL_PATH = './model/env_model/'

device = 'cpu'

# -----------------------给损失函数添加正则化项-----------------------------------------

def training_loop_l2reg(n_epochs:int, optimizer, model, loss_fn, train_loader):
    l2_lambda = 0.001
    for epoch in range(1, n_epochs +1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            
            # l2_norm = sum(p.pow(2.0).sum()
            #             for p in model.parameters())
            
            # loss = loss + l2_lambda * l2_norm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training Loss {}'.format(datetime.datetime.now(),
                                                        epoch, 
                                                        loss_train/len(train_loader)))
            



# # ----------------------构造残差块，并构建非常深的模型----------------------------------
# class ResBlock(nn.Module):
#     def __init__(self, n_input, n_output) -> None:
#         super().__init__()
#         # self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
#         # self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
#         # # 使用了特殊的初始化方法，对卷积层的权重进行初始化，对批量归一层的权重和偏置进行初始化
#         # torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
#         # torch.nn.constant_(self.batch_norm.weight, 0.5)
#         # torch.nn.init.zeros_(self.batch_norm.bias)
#         self.fc1 = nn.Linear(input, 32)
#         self.bac


#         #--------------------------------------------------------------
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.batch_norm(x)
#         out = torch.relu(out)
#         return out + x

class NetResDeep(nn.Module):
    def __init__(self, n_input=9, n_output=7) -> None:
        # 6个observations, 3个action，1个reward
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        # self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        # self.resblock = nn.Sequential(*(n_block * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(n_input, 256)
        # self.batchnorm1 = nn.BatchNorm1d(num_features=64)
        self.fc2 = nn.Linear(256, 256)
        # self.batchnorm2 = nn.BatchNorm1d(num_features=32)
        self.fc3 = nn.Linear(256,128)
        # self.batchnorm3 = nn.BatchNorm1d(num_features=16)
        self.fc4 = nn.Linear(128,n_output)

    def forward(self, x):
        # out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        # out = self.resblock(out)
        # out = F.max_pool2d(out, 2)
        # out = out.view(-1, 8*8*self.n_chans1)
        
        out = torch.relu(self.fc1(x))
        # out = self.batchnorm1(out)
        out = torch.relu(self.fc2(out))
        # out = self.batchnorm2(out)
        out = torch.relu(self.fc3(out))
        # out = self.batchnorm3(out)
        out = self.fc4(out)
        return out


if __name__ == "__main__":
    # env model training, model output dir: ./

    using_neural_network = False

    env_data = pd.read_csv('./output/training/env_model_data/env_data.csv')
    env_data = np.array(env_data)


    env_data = env_data.astype(np.float32)
    data_x = env_data[:, 0:9]
    data_y = env_data[:, 9:16]

    normalizeHelper = NormalizeHelper(method='StdScaler')
    
    data_x = normalizeHelper.fit(data_x)

    if using_neural_network:
        learning_rate = 1e-1
        
        model = NetResDeep()
        loss_fn = nn.MSELoss()
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loader = DataLoader(dataset=[data_x, data_y],
                            batch_size=128,
                            shuffle=True,
                            using_pytorch=True)

        training_loop_l2reg(n_epochs=int(5e1),
                            optimizer=optimizer,
                            model=model,
                            loss_fn=loss_fn,
                            train_loader=loader)
        

        print('original: ', torch.tensor(data_y[0:10, :], dtype=torch.float32))
        with torch.no_grad():
            input_x = torch.tensor(normalizeHelper.normalize(data_x[0:10, :]), dtype=torch.float32)
            print('predict -  original: ', model(input_x) - torch.tensor(data_y[0:10, :], dtype=torch.float32))
    else:
        # 随机森林预测
        regressor = RandomForestRegressor()
        regressor.fit(data_x, data_y)

        print('original: ', data_y[0:10, :])
        predict_y = regressor.predict(data_x[0:10,:])
        print('predict -  original: ',  predict_y - data_y[0:10, :])
        print('error %: ', (predict_y - data_y[0:10, :]) / data_y[0:10, :])
        
        with open(ENV_MODEL_PATH + 'regressor.pickle','wb') as f: 
            pickle.dump(regressor,f) #将训练好的模型clf存储在变量f中，且保存到本地

    pass
