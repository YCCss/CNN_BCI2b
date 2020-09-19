# coding = utf-8

import numpy as np 
import pandas as pd 
import torch 
import datetime
import matplotlib.pyplot as plt
from torch import nn
from torchkeras import summary
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
import read_data as rd
import torchvision.models as mo

resnet18 = mo.resnet18()

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18,self).__init__()
        self.resnet18 = resnet18
        self.linear1 = nn.Linear(1000,128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.resnet18(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
'''
net = Resnet18()
summary(net,input_shape= (3,30,30))
print(net)
while 1:
    pass
'''

class LetNet(nn.Module):
    def __init__(self):
        super(LetNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y

'''
a = np.arange(2700*2)
a = a.reshape(-1,3,30,30)
a = torch.tensor(a,dtype = torch.float32)
m = LetNet()
print(m(a))
print(m.forward(a))'''

#summary(net,input_shape= (3,30,30))
#print(net)
'''
def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
'''
def train_step(model,features,labels):
    model.train()
    model.optimizer.zero_grad()
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)
    loss.backward()
    model.optimizer.step()
    return loss.item(),metric.item()

def test_step(model,features,labels):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions,labels)
        metric = model.metric_func(predictions,labels)
    return loss.item(),metric.item()

def train_model(model,epochs,train,test,log_step_freq):
    metric_name = model.metric_name
    
    dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)
    
    for epoch in range(1,epochs+1):
        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        for step,(features,labels) in enumerate(train,1):
            loss,metric = train_step(model,features,labels)
         
            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                        (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(test, 1):

            val_loss,val_metric = test_step(model,features,labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric   

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step, 
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
              "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)

    print('Finished Training...')
    
    return dfhistory

def predict(model,dl):
    model.eval()
    with torch.no_grad():
        result = model.forward(dl)
    #返回概率值
    return(result.data)
def cal_one(y_pred_probs):
    y_pred = torch.where(y_pred_probs>0.5,
        torch.ones_like(y_pred_probs),torch.zeros_like(y_pred_probs))
    return y_pred

if __name__ == '__main__':
    file_path = 'data/bci2b/subject1.mat'
    #读取数据
    train_x,train_y,test_x,test_y = rd.read_data_torch(file_path)

    #数据批处理
    train = TensorDataset(torch.tensor(train_x),torch.tensor(train_y))
    test = TensorDataset(torch.tensor(test_x),torch.tensor(test_y))
    train,test = DataLoader(train,batch_size = 40),DataLoader(test,batch_size = 40)

    model = LetNet()
    #model = Resnet18()
    print(model)

    model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
    model.loss_func = torch.nn.BCELoss()
    model.metric_func = lambda y_pred,y_true: roc_auc_score(y_true.data.numpy(),y_pred.data.numpy())
    model.metric_name = "auc"

    epochs = 200

    dfhistory = train_model(model,epochs,train,test,log_step_freq = 5)
    dfhistory.to_csv('./data/log.csv')
    #plot_metric(dfhistory, 'loss')
    #plot_metric(dfhistory, 'auc')

    #保存模型
    torch.save(model.state_dict(), "./data/model_parameter.pkl")
    #print(dfhistory)


    #features,labels = next(iter(train))
    #print(train_step(model,features,labels))



    p = predict(model,torch.tensor(train_x[0:10]))
    print(p)
    p = cal_one(p)
    print(p)


