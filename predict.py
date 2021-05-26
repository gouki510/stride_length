from pandas.core.tools.datetimes import DatetimeScalar, DatetimeScalarOrArrayConvertible
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.nn.modules import padding
from train import Regression
import openpyxl
import glob
from natsort import natsorted

#データをpytorchのtensorに変換
path2data = '/home/mech-user/Desktop/3S/datascience/Regression/data/test/dataset/left'
listofdata = natsorted(glob.glob(path2data + '/*csv'))
datas = []
for i,data in enumerate(listofdata):
    data_path = os.path.join(path2data,data)
    # dataとlabelに分ける
    df = pd.read_csv(data_path)
    d = df.loc[:,'accelX':'gyroZ'].values
    print(data)
    print(d.shape)
    datas.append(d)

#trainデータの量の確認

#for data in datas:
    
#6channel 高さ1 幅20 にする
inputs = torch.stack([torch.from_numpy(data).reshape(6,1,20) for data in datas])
#labels = torch.stack([torch.tensor(label) for label in labels])

#input sizeの確認
print(inputs[0].shape)


model = Regression()
model.load_state_dict(torch.load("model/left.pt"))
model.eval()

print(type(inputs))
output = model(inputs.float())
output = output.detach().numpy()
output = pd.DataFrame(output)
#print(output)
#print(output.shape)
print(output)
df = pd.read_csv("/home/mech-user/Desktop/3S/datascience/Regression/data/test/target/left_cw1.csv")
print(df)
print(df.columns)
#print(pd.merge(output,df))
df1 = pd.concat([df, output.set_index(df.index)], axis=1)
print(df1)

df1.to_csv('infer/left/infer_left2.csv')