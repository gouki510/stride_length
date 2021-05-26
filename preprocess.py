import pandas as pd
import numpy as np
import os
import glob
import datetime
import time

train_path = '/home/mech-user/Desktop/3S/datascience/Regression/data/training'
target_path = '/home/mech-user/Desktop/3S/datascience/Regression/data/target'
test_path = '/home/mech-user/Desktop/3S/datascience/Regression/data/test/data'
target_test_path = '/home/mech-user/Desktop/3S/datascience/Regression/data/test/target'

#train
def train():
    listoftrain = sorted(glob.glob(train_path + '/*csv'))
    listoftarget = sorted(glob.glob(target_path + '/*csv'))
    target_idxs = []
    idx = []
    idxs = []
    for train,target in zip(listoftrain,listoftarget):
        train_df = pd.read_csv(train, index_col=0)
        target_df = pd.read_csv(target, names=['time', 'stride'],index_col=0)
        for j,time in enumerate(target_df.index.tolist()):
            for i,date in enumerate(train_df.index.tolist()):
                if datetime.datetime.strptime(time,'%H:%M:%S') == datetime.datetime.strptime(date[11:19],'%H:%M:%S'):
                    idx.append(date)
            target_idxs.append(time)        
            idxs.append(idx)    
            idx = []
        for i,(target_id,id) in enumerate(zip(target_idxs,idxs)):
            if len(id) == 20 and type(target_df.loc[target_id,'stride']) == np.float64 :
                df = train_df.loc[id].assign(stride=target_df.loc[target_id,'stride'])
                print(df.shape)
                df.to_csv('dataset2/left/' + "data" + str(i) + os.path.basename(train))
            elif len(id) == 0:
                pass
            elif len(id) == 21:
                df = train_df.loc[id].assign(stride=target_df.loc[target_id,'stride'])
                df = df.drop(df.index[-1])
                print("21")
                print(df.shape)
                df.to_csv('dataset2/left/' + "data" + str(i) + os.path.basename(train))
            else :
                df  = train_df.loc[id].assign(stride=target_df.loc[target_id,'stride'])
                while True:
                    df  = df.append(train_df.loc[id[-1]])
                    print("19")
                    print(df.shape)
                    if len(df.index) == 20:
                        break
                df.to_csv('dataset2/left/' + "data" + str(i) + os.path.basename(train))
        target_idxs = []

        target_idxs = []
        idxs = []

#test
def test():
    listoftest = sorted(glob.glob(test_path + '/*csv'))
    listoftarget = sorted(glob.glob(target_test_path + '/*csv'))
    target_idxs = []
    idx = []
    idxs = []
    for test,target in zip(listoftest,listoftarget[1:]):
        test_df = pd.read_csv(test, index_col=0)
        target_df = pd.read_csv(target, names=['time', 'stride'],index_col=0)
        for j,time in enumerate(target_df.index.tolist()):
            for i,date in enumerate(test_df.index.tolist()):
                if datetime.datetime.strptime(time,'%H:%M:%S') == datetime.datetime.strptime(date[11:19],'%H:%M:%S'):
                    idx.append(date)
            target_idxs.append(time)        
            idxs.append(idx)    
            idx = []
        for i,(target_id,id) in enumerate(zip(target_idxs,idxs)):
            
            if len(id) == 20 and type(target_df.loc[target_id,'stride']) == np.float64:
                df = test_df.loc[id].assign(stride=target_df.loc[target_id,'stride'])
                print("20")
                print(df.shape)
                df.to_csv('/home/mech-user/Desktop/3S/datascience/Regression/data/test/dataset/right/' + "data" + str(i) + os.path.basename(test))
            elif len(id) == 0:
                pass
            elif len(id) == 21:
                df = test_df.loc[id].assign(stride=target_df.loc[target_id,'stride'])
                df = df.drop(df.index[-1])
                print("21")
                print(df.shape)
                
                df.to_csv('/home/mech-user/Desktop/3S/datascience/Regression/data/test/dataset/right/' + "data" + str(i) + os.path.basename(test))
            else :
                df  = test_df.loc[id].assign(stride=target_df.loc[target_id,'stride'])
                while True:
                    df  = df.append(test_df.loc[id[-1]])
                    print("19")
                    print(df.shape)
                    if len(df.index) == 20:
                        break
                df.to_csv('/home/mech-user/Desktop/3S/datascience/Regression/data/test/dataset/right/' + "data" + str(i) + os.path.basename(test))
        target_idxs = []
        idxs = []

if __name__ == "__main__":
    train()


