"""
Reads preprocessed data

"""
from config import get_config
import numpy as np
from utils import *
import pandas as pd

def readdata(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/train.hdf5')
    testlabelData= ddio.load('dataset/trainlabel.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    valData = ddio.load('dataset/test.hdf5')
    vallabelData= ddio.load('dataset/testlabel.hdf5')
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    return (X, y, Xval, yval)

def readdata_nosplit(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata_std.hdf5')
    testlabelData= ddio.load('dataset/labeldata_std.hdf5')
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    return (X, y)

def readdata_nosplit_scaled(input_size, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata_scaled.hdf5')
    testlabelData= ddio.load('dataset/labeldata_scaled.hdf5')
    indexData= ddio.load('dataset/index_scaled.hdf5')

    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    #np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    subjectLabel = (np.array(pd.DataFrame(indexData)[1]))
    nums = ['100','101','103','105','106','107','108','109','111','112','113','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
    num_index = 0
    group = []
    for x in subjectLabel:
        for beat in range(x):    
            group.append(nums[num_index])
        num_index += 1
    #group = np.array(group)
    return (X, y, group)

def readdata_nosplit_scaled_subject(input_size, subjects, feature):
    import deepdish.io as ddio
    mkdir_recursive('dataset')
    trainData = ddio.load('dataset/targetdata_scaled.hdf5')
    testlabelData= ddio.load('dataset/labeldata_scaled.hdf5')
    indexData= ddio.load('dataset/index_scaled.hdf5')

    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    att = np.concatenate((X,y), axis=1)
    #np.random.shuffle(att)
    X , y = att[:,:input_size], att[:, input_size:]
    subjectLabel = (np.array(pd.DataFrame(indexData)[1]))
    print("==============")
    print(subjectLabel)
    nums = ['100','101','103','105','106','107','108','109','111','112','113','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
    num_index = 0
    group = []
    for x in subjectLabel:
        for beat in range(x):    
            group.append(nums[num_index])
        num_index += 1
    #group = np.array(group)
    return (X, y, group)

def main(config):
    classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
    #print('feature:', config.feature)
    #np.random.seed(0)
    if config.split == True:
        (X,y, Xval, yval) = readdata(config.input_size, config.feature)
        print(X.shape)
        print(y.shape)
        print(pd.DataFrame(y, columns=classes).sum())
        print(Xval.shape)
        print(yval.shape)
        print(pd.DataFrame(yval, columns=classes).sum())


    else:
        (X,y, group) = readdata_nosplit_scaled(config.input_size, config.feature)
        print(X.shape)
        print(y.shape)
        df = pd.DataFrame(y, columns=classes)
        #print(df)
        print(df.sum())
        #print(group)
        #print(np.unique(group))
        #print(np.unique(group, return_counts = True))
        subject, subject_count = np.unique(group, return_counts = True)
        print(subject_count)
        marker = 0
        beat_table = pd.DataFrame()
        for c in subject_count:
            beat_table = pd.concat([beat_table, pd.DataFrame(y[marker:marker+c], columns=classes).sum()], axis=1)
            marker = marker+c
        beat_table.columns = subject
        beat_table = beat_table.T
        print(beat_table)



if __name__=="__main__":
    config = get_config()
    main(config)
