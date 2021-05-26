from pandas.core.tools.datetimes import DatetimeScalar, DatetimeScalarOrArrayConvertible
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.nn.modules import padding
import copy


#データをpytorchのtensorに変換
def preprocess():
    path2data = '/home/mech-user/Desktop/3S/datascience/Regression/dataset/left'
    listofdata = os.listdir(path2data)
    labels = []
    datas = []
    for data in listofdata:
        data_path = os.path.join(path2data,data)
        # dataとlabelに分ける
        df = pd.read_csv(data_path)
        labels.append(df['stride'][0])
        d = df.loc[:,'accelX':'gyroZ'].values
        datas.append(d)

    #trainデータの量の確認
    #print(len(datas))

    #6channel 高さ1 幅20 にする
    inputs = torch.stack([torch.from_numpy(data).reshape(6,1,20) for data in datas])
    labels = torch.stack([torch.tensor(label) for label in labels])

    #input sizeの確認
    #print(inputs[0].shape)

    #dataをloadする
    train_dataset = torch.utils.data.TensorDataset(inputs, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True)

#modelを定義
class Regression(nn.Module):
    def __init__(self) :
        super(Regression,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=1)
        )

        self.fc1 = nn.Linear(120,60)
        self.fc2 = nn.Linear(60,1)

    def forward(self,x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        flatten = nn.Flatten()
        x = flatten(x)
        #print(x.shape)
        hidden = self.fc1(x)
        x = self.fc2(hidden)

        return x
    
net = Regression()

#lossとoptimizerを定義
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

#学習
def train():
    train_acc_list = []
    train_loss_list = []

    nb_epoch = 150

    for epoch in range(nb_epoch):
        train_loss = 0
        train_acc = 0

        #train
        net.train()
        for i, (data,labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.float()
            labels = labels.float()
            outputs = net(data)
            loss = criterion(outputs,labels)
            train_loss += loss.item()
            #train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        print('Epoch [{}/{}], loss: {loss:.4f} train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}'
                    .format(epoch+1, nb_epoch, i+1, loss=avg_train_loss, train_loss=avg_train_loss, train_acc=avg_train_acc))

    model_path = 'model/left.pt'
    torch.save(net.state_dict(), model_path)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,6))
    plt.plot(train_loss_list,lw=3,c='b')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('regression')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.show()

if __name__ == "__main__":
    train()