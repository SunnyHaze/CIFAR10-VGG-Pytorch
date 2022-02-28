from cProfile import label
import torch
import numpy as np
with open('Cifar-10_Unpacked/Label2Name.npy','rb') as f:
    diclabel2name = np.load(f) 

# 原始数据是3072维向量，转化成可以plt的显示的张量（32,32,3）
def TransTensor2Img(raw):
    r,g,b = torch.split(raw,1024)
    r = r.reshape(1,32,32)
    g = g.reshape(1,32,32)
    b = b.reshape(1,32,32)
    return torch.cat((r,g,b), dim=0)

def oneHot2label(t:torch.Tensor):
    for i in range(10):
        if(t[i] == 1):
            return i

def label2name(i:int):
    return diclabel2name[i]