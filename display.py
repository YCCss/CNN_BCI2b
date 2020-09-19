# coding = utf-8

import matplotlib.pyplot as plt
import pandas as pd

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

#保存数据
#dfhistory.to_csv('./data/log.csv')
#加载数据
dfhistory = pd.read_csv('./data/log.csv')

plot_metric(dfhistory, 'loss')
plot_metric(dfhistory, 'auc')