
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import torchvision
train = datasets.MNIST(root='../', train=True, transform=transforms.ToTensor(), download=True)
test = datasets.MNIST(root='../', train=False, transform=transforms.ToTensor(), download=False)
