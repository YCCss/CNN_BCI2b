# coding = utf-8

import train 
import torch
import read_data as rd

file_path = 'data/bci2b/subject1.mat'
#读取数据
train_x,train_y,test_x,test_y = rd.read_data_torch(file_path)

net_clone = train.LetNet()
#模型读取预测
net_clone.load_state_dict(torch.load("./data/model_parameter.pkl"))

p = train.predict(net_clone,torch.tensor(train_x[0:10]))
print(p)
p = train.cal_one(p)
print(p)
print(train_y[0:10])
