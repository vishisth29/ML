import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import torchvision


class Generator(torch.nn.Module):
    def __init__(self, g_input, g_output):
        self.f1 = nn.Linear(g_input, 256)
        self.f2 = nn.Linear(self.f1.out_f)
        self.f3 = nn.Linear(self.f2.out_f)
        self.f4 = nn.Linear(self.f3.out_f, g_output)

    def forward(self,g):
        g=nn.leaky_relu(self,)
        g=nn.leaky_relu(self.f2(g), 0.2)
        g=nn.leaky_relu(self.f3(g), 0.2)
        return torch.tanh(self.f4(g))



class Discriminator(torch.nn.Module):
    def __init__(self,x):
        tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.maximum(h1, alpha*h1)
        log = tf.layers.dense(h1, 1, activation=None)
        o = tf.nn.sigmoid(log)
        return o, log

g=Generator(g_input, g_output).to(device)
d=Discriminator(mnist_data).to(device)
def train(x):

epochs = 200
for epochs in range(1, epochs+1):           
    dLoss, gLoss = [], []
    for index,(x, _) in enumerate(train_samples):
        dLoss.append(train(x))
        gLoss.append(train(x))





