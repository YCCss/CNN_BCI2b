import scipy.io as si
import matplotlib.pyplot as plt
import numpy as np


def read_data_torch(file_path):
    data = si.loadmat(file_path)
    
    #训练数据处理
    #先把训练数据展开，400000*3，然后取每一列,reshape成400*1000，
    #取每一行的900个数据，然后将三个通道的数据水平堆叠
    
    train_x = data['train_data']
    train_x = np.array(train_x,dtype=np.float32)
    n1 = train_x[...,0]#1列
    n2 = train_x[...,1]#2列
    n3 = train_x[...,2]#3列
    n1 = n1.reshape([-1,1000])[...,0:900]
    n2 = n2.reshape([-1,1000])[...,0:900]
    n3 = n3.reshape([-1,1000])[...,0:900]
    #水平堆叠
    train_x = np.hstack([n1,n2,n3]) #400*2700
    #print(train_x.shape)
    train_x = train_x.reshape(-1,3,30,30)
    #print(train_x.shape) #400*3*30*30

    #训练标签处理
    train_y_ = data['train_label']
    train_y_ = np.array(train_y_,dtype=np.float32)
    train_y = train_y_-1
    #print(train_y)


    #测试数据处理
    test_x = data['test_data']
    test_x = np.array(test_x,dtype=np.float32)
    n1 = test_x[...,0]#1列
    n2 = test_x[...,1]#2列
    n3 = test_x[...,2]#3列
    n1 = n1.reshape([-1,1000])[...,0:900]
    n2 = n2.reshape([-1,1000])[...,0:900]
    n3 = n3.reshape([-1,1000])[...,0:900]
    #水平堆叠
    test_x = np.hstack([n1,n2,n3])
    test_x = test_x.reshape(-1,3,30,30)
    
    #测试标签处理
    test_y_ = data['test_label']
    test_y_ = np.array(test_y_,dtype=np.float32)
    test_y = test_y_-1
    
    #(400,3,30,30) (400, 1) (320,3,30,30) (320, 1)
    return train_x,train_y,test_x,test_y

'''
file_path = 'data/bci2b/subject1.mat'
train_x,train_y,test_x,test_y = read_data_torch(file_path)
#print(train_x.dtype,train_y.shape,test_x.shape,test_y.shape)
print(train_y[0:10])'''