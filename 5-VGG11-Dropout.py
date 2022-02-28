import os
from matplotlib.pyplot import imshow
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import conv2d, nn, sigmoid, tensor
import numpy as np
from imports.ParametersManager import *
from imports.utlis import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

# 超参数
MODELNAME='VGG11'
MODELFILEDIR = 'PretrainedModels' # 模型参数存储路径
BatchSize = 128
LEARNINGRATE = 0.01
epochNums = 10
SaveModelEveryNEpoch = 2 # 每执行多少次保存一个模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 构建模型参数文件存取路径
if not os.path.exists(MODELFILEDIR):
    os.mkdir(MODELFILEDIR)
MODELFILEPATH = os.path.join(MODELFILEDIR, MODELNAME+'_model.pt')


# 可以将数据线包装为Dataset，然后传入DataLoader中取样
class MyDataset(Dataset):
    def __init__(self,SetType) -> None:
        with open(SetType + 'Images.npy','rb') as f:
            self.images =torch.tensor(np.load(f), dtype=torch.float32)
            self.images = (self.images - 0.5) / 0.5 # 将数据范围映射到[-1,1]很重要，可以提高准确率
        with open(SetType + 'Labels.npy','rb') as f:
            tmp = np.load(f)
            print(tmp)
        self.labels=[]
        for num in tmp:
             self.labels.append([1 if x == num else 0 for x in range(10)])
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return len(self.labels)


def blockVGG(covLayerNum,inputChannel, outputChannel, kernelSize, withFinalCov1:bool):
    layer = nn.Sequential()
    layer.add_module('conv2D1',nn.Conv2d(inputChannel, outputChannel,kernelSize,padding=1))
    layer.add_module('relu-1',nn.ReLU())
    for i in range(covLayerNum - 1):
        layer.add_module('conv2D{}'.format(i),nn.Conv2d(outputChannel, outputChannel,kernelSize,padding=1))
        layer.add_module('relu{}'.format(i),nn.ReLU())
    if withFinalCov1:
        layer.add_module('Conv2dOne',nn.Conv2d(outputChannel,outputChannel, 1))
    layer.add_module('max-pool',nn.MaxPool2d(2,2))
    return layer

# 定义网络结构
class VGG11(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = blockVGG(1,3,64,3,False)
        
        self.layer2 = blockVGG(1,64,128,3,False)
        
        self.layer3 = blockVGG(2,128,256,3,False)
        
        self.layer4 = blockVGG(2,256,512,3,False)
        
        self.layer5 = blockVGG(2,512,512,3,False)
        self.layer6 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # this may work in solving over-fit but no improvement in my test
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Dropout(0.5), # this may work in solving over-fit but no improvement in my test
            nn.Linear(100,10),
            # nn.ReLU(),
            # nn.Softmax(1)
        )
    '''
        You can try to put some dropout layer between the Conv layer. This really works when you desire to imporve the performance in the test set.
        
        But it may makes the model hard to fitting, you can use the model without dropout first, and put in some dropout layer after it start to fit.(When have an ACC of 30%) 
    '''
    def forward(self,x:torch.Tensor):
        x = self.layer1(x) # 执行卷积神经网络部分
        x = self.layer2(x) # 执行全连接部分
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)
        x = self.layer6(x)
        return x

# 定义准确率函数
def accuracy(output , label):
    rightNum = torch.sum(torch.max(label,1)[1].eq(torch.max(output,1)[1]))
    return rightNum / len(label)
        
if __name__ == "__main__":    
    # 模型实例化        
    model = VGG11()
    print(model)
    # # 如果有“半成品”则导入参数
    parManager = ParametersManager(device)
    if os.path.exists(MODELFILEDIR):
        parManager.loadFromFile(MODELFILEDIR)
        parManager.setModelParameters(model)
    else:
        print('===No pre-trained model found!===')

    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNINGRATE)
    
    dirOfDataset = 'Cifar-10_Unpacked/'
    # 构建训练集
    TrainDataset = MyDataset(dirOfDataset + 'Train')
    # 构建测试集
    TestDataset = MyDataset(dirOfDataset + 'Test')
    # 构建训练集读取器
    TrainLoader = DataLoader(TrainDataset,num_workers=8, pin_memory=True, batch_size=BatchSize, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(TrainDataset))))
    # 构建测试集读取器：
    TestLoader = DataLoader(TestDataset,num_workers=8, pin_memory=True, batch_size=BatchSize, sampler= torch.utils.data.sampler.SubsetRandomSampler(range(len(TestDataset))))
    # 
    print('len(TrainLoader):{}'.format(len(TrainLoader)))
    
    # 检查分割是否正确的函数，分为两行，以行为顺序排列和输出结果一一对应
    def testLoader():
        inputs, classes = next(iter(TrainLoader))
        inputs = inputs[:10]
        classes = classes[:10]
        print(inputs.shape)
        print(classes.shape)
        print(classes) # 查看标签
        for i in range(len(inputs)):
            plt.subplot(2,5,i+1)
            img = torch.transpose(inputs[i], 0,2)
            img = torch.transpose(img,1,0)
            plt.imshow(img)
            plt.title(label2name(oneHot2label(classes[i])))
        plt.show()
        
    # testLoader()

    TrainACC = []
    TestACC = []
    GlobalLoss = []
    for epoch in range(epochNums):
        print("===开始本轮的Epoch {} == 总计是Epoch {}===".format(epoch, parManager.EpochDone))
        
        # 收集训练参数
        epochAccuracy = []
        epochLoss = []
        model.train()
        #=============实际训练流程=================
        for batch_id, (inputs,label) in enumerate(TrainLoader):
            # torch.train()
            # 先初始化梯度0
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = criterion(output,label.cuda())
            loss.backward()
            optimizer.step()
            epochAccuracy.append(accuracy(output,label.cuda()).cpu())
            epochLoss.append(loss.item()) # 需要获取数值来转换
            if batch_id % (int(len(TrainLoader) / 20)) == 0: 
                print("    当前运行到[{}/{}], 目前Epoch准确率为：{:.2f}%，Loss：{:.6f}".format(batch_id,len(TrainLoader), np.mean(epochAccuracy) * 100, loss))
        #==============本轮训练结束==============
        # 收集训练集准确率
        TrainACC.append(np.mean(epochAccuracy)) 
        GlobalLoss.append(np.mean(epochLoss))
        # ==========进行一次验证集测试============
        localTestACC = []
        model.eval() # 进入评估模式，节约开销
        for inputs, label in TestLoader:
            torch.no_grad() # 上下文管理器，此部分内不会追踪梯度/
            output = model(inputs.cuda())
            localTestACC.append(accuracy(output,label.cuda()).cpu())
        # ==========验证集测试结束================
        # 收集验证集准确率
        TestACC.append(np.mean(localTestACC))
        print("当前Epoch结束，训练集准确率为：{:3f}%，测试集准确率为：{:3f}%".format(TrainACC[-1] * 100, TestACC[-1] * 100))
        # 暂存结果到参数管理器
        parManager.oneEpochDone(LEARNINGRATE,TrainACC[-1],TestACC[-1],GlobalLoss[-1])
        # 周期性保存结果到文件
        if epoch == epochNums - 1 or epoch % SaveModelEveryNEpoch == 0:
            parManager.loadModelParameters(model)
            parManager.saveToFile(MODELFILEDIR)
            
    # 查看此次训练之后结果
    parManager.show()
    # 绘图
    plt.figure(figsize=(10,7))
    plt.subplots_adjust(left=0.1,bottom=0.1,top=0.9,right=0.9,wspace=0.1,hspace=0.3)
    plt.subplot(2,1,1)
    plt.plot(range(parManager.EpochDone),parManager.TrainACC,marker='*' ,color='r',label='Train')
    plt.plot(range(parManager.EpochDone),parManager.TestACC,marker='*' ,color='b',label='Test')

    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.title("{} on Cifar-10".format(MODELNAME))
    plt.text(int(parManager.EpochDone *0.8),0.5,'Train ACC: {:.6f}\nTest ACC: {:.6f}\nEpoch:{}'.format(parManager.TrainACC[-1],parManager.TestACC[-1], parManager.EpochDone))
    plt.subplot(2,1,2)
    plt.title('Learning Rates')
    plt.xlabel('Epoch')
    plt.ylabel('$log_{10}$(Learning Rates)')
    plt.ylim(0,-5)
    plt.plot([x for x in range(parManager.EpochDone)], np.log(parManager.LearningRate) / np.log(10))
    plt.savefig('Train-{}-{}Epoch.jpg'.format(MODELNAME,parManager.EpochDone))
    plt.show()