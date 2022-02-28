import pickle
import os
import numpy as np
from imports.utlis import *
from torchvision.datasets import CIFAR10
import torch
# the path where raw datasets are saved
baseDir = './cifar-10-batches-py'
# Download the datasets with the pytoch API
CIFAR10('./', train=True, download=True)
CIFAR10('./', train=True, download=True)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# get the true dictionary between the ID and the real name of the label.
def getLabel2Name():
    Labeldict = unpickle(os.path.join(baseDir,'batches.meta'))[b'label_names']
    num2Label = []
    for i in Labeldict:
        num2Label.append(i.decode('utf-8') )
    return num2Label

# get labels and the data from a data_batch file, indexed by parameter ID.
def decodeABatch(filename):
    path = os.path.join(baseDir, filename)
    batch = unpickle(path)
    labels = torch.tensor(batch[b'labels'])
    data = torch.tensor(batch[b'data'] ,dtype=float)/255     
    BatchLen = len(data)
    finalData = torch.zeros((BatchLen,3,32,32))
    for id,m in enumerate(data):
        finalData[id] = TransTensor2Img(m)
    return labels, finalData

# Decode the Trainning data to *.npy file, Memory friendly but disk unfriendly (larger disk space are taken).
def GetTrainSet():
    TrainingSize = 50000
    batchlen = 10000
    labels = torch.zeros((TrainingSize))
    data = torch.zeros((TrainingSize,3 ,32,32))
    for i in range(5):
        print("decoding Patch {}...".format(i))
        batchLabels,batchData = decodeABatch( "data_batch_" + str(i + 1))
        labels[i * batchlen : (i + 1) * batchlen] = batchLabels
        data[i * batchlen : (i + 1) * batchlen] = batchData
    return labels, data
# Decode the test set
def GetTestSet():
    return decodeABatch('test_batch')
# save them to 'Cifar-10_Unpacked/' 
# then it can easily to be read by Neural network models.
def saveDataSets():
    baseDatasetDir = 'Cifar-10_Unpacked/'
    if not os.path.exists(baseDatasetDir):
       os.makedirs(baseDatasetDir)
    print('Re-forming TraingSet...')
    TrainY, TrainX = GetTrainSet()
    # Training data
    with open(baseDatasetDir + 'TrainImages.npy','wb') as f:
        np.save(f,TrainX)
    with open(baseDatasetDir + 'TrainLabels.npy','wb') as f:
        np.save(f,TrainY)    
    print('Done!')
    print('Reforming Test set...')
    TestY, TestX = GetTestSet()
    # Testing Set
    with open(baseDatasetDir + 'TestImages.npy','wb') as f:
        np.save(f,TestX)
    with open(baseDatasetDir + 'TestLabels.npy','wb') as f:
        np.save(f,TestY)
    print('Done!')
    # num2Label dict
    with open(baseDatasetDir + 'Label2Name.npy','wb') as f:
        np.save(f, getLabel2Name())
        
saveDataSets()