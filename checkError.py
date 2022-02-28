from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
transform = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
         ])
print(CIFAR10)

TrainDataset = CIFAR10('./data', train=True, transform=transform, download=True)

TrainLoader = DataLoader(TrainDataset,num_workers=1, pin_memory=True, batch_size=10)
a = None
b = None
if __name__ == '__main__':
    for i,b in TrainLoader:
        print(i.shape)
        print(i[0])
        a = i[0]
        break
    with open('Cifar-10_Unpacked/Train'+'Images.npy','rb') as f:
        images =torch.tensor(np.load(f), dtype=torch.float32)
        print(images.shape)
        print(images[0])
        b = images[0]
        c = (b-0.5)/ 0.5
        print(a - c)