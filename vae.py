import torch
import numpy
import torch.nn as nn
from torch import nn
from torch.autograd.variable import Variable
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
train = datasets.MNIST(root='../', train=True, transform=transforms.ToTensor(), download=True)
test = datasets.MNIST(root='../', train=False, transform=transforms.ToTensor(), download=False)


class VAE(nn.Module):
    def __init__(self, xdimen, y1dimen, y1dimen, zdimen ):
        self(VAE, self).__init__()

        self.transform1 = nn.Linear(784,512)
        self.transform2 = nn.Linear(256,20)
        self.transform3 = nn.Linear(512,20)
        self.transform4 = nn.Linear(256,20)

    def encoder(self, h1):
        h1=self.transform12(h1)
        h1=self.transform11(h1)
        h1=self.transform21(h1)
        h1=self.transform22(h1)
        h1=self.transform31(h1)

        temp=torch.exp(self.transform32(h1))
        return temp

    self.transform32 = nn.Linear(256, 20)
    self.transform11 = nn.Linear(784,512)
    self.transform22 = nn.Linear(256,20)
    self.transform12 = nn.Linear(512,20)
    self.transform31 = nn.Linear(256,20)

    def decoder(self,h2):
        h2=self.transform12(h1)
        h2=self.transform11(h1)
        h2=self.transform21(h1)
        h2=self.transform22(h1)
        h2=self.transform31(h1)

        return h2;

    def forward(self, x):
        mu,v=self.encoder(x)
        z = self.decoder(mu, l)
        return self.sampling(z), mu,l

vae=VAE()

optimizer = optim.Adam(vae.parameters())

def train(epoch):
    vae.train()
    loss=0
    for epochs in range(200):
    for batch, (batchdata, _) in enumerate(data_count):
        optimizer.zero_grad()
        rec, mu, log = vae(data)
        loss = loss_function(rec, mu, log)
        loss.backward()
        vae_opt.step()
        print('epoch = {}'.format(epochs))


def test(epoch):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in testl:
            data = data.cuda()
            rec, mu, log = vae(data)
            test_loss += loss_func(rec, mu, log).item()

    testl /= len(testl.dataset)
    print('====> Test loss is : {:.4f}'.format(testl))

for epoch in range(50):
    test_temp=train(epoch)
    test()
