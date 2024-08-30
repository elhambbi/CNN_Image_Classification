import torch
from torch import nn
import torchvision
from torchvision import datasets   # to impot FashionMNist images
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm   # for progress bar
from timeit import default_timer as timer 
import random
import os

print(torch.__version__)
print(torchvision.__version__)


# download the first time
data_dir = "data"
train_dataset = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # do we want the training dataset?
    download=True, # do we want to download yes/no?
    transform=ToTensor(), # how do we want to transform the data?
    target_transform=None #transform the labels/targets?
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

print(len(train_dataset), len(test_dataset))

# save the dataset for next usages
torch.save(train_dataset, os.path.join(data_dir, 'fashion_mnist_train.pt'))
torch.save(test_dataset, os.path.join(data_dir, 'fashion_mnist_test.pt'))