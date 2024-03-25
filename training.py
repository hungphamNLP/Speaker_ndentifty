import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from model import *
import tqdm
import time

from datasets import *


def test(model, device, test_loader):
    model.eval()
    accurate_labels = 0
    all_labels = 0
    losses = []
    with torch.no_grad():
        for batch_idx, (data0, data1, label) in enumerate(tqdm.tqdm(test_loader)):
            data0, data1, label = data0.to(device), data1.to(device), label.to(device)
            out = model(data0, data1)
            loss_function = criterion(out, label)
            losses.append(loss_function.item())

            accurate_labels += torch.sum(torch.argmax(out, dim=1) == label).cpu()
            all_labels += len(label)
    test_loss = np.mean(losses)
    test_accuracy = 100. * accurate_labels / all_labels
    print('\nTest set: Average loss = {:.4f}, Test Accuracy = {:.4f}\n'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy


def train(model, device, train_loader, epoch):
    model.train()
    losses = []
    accurate_labels = 0
    all_labels = 0
    len_train_loader = len(train_loader)
    for batch_idx, (data0, data1, label) in enumerate(tqdm.tqdm(train_loader)): #tqdm.tqdm(train_loader)
        data0, data1, label = data0.to(device), data1.to(device), label.to(device)
        optimizer.zero_grad()
        
        out = model(data0, data1)
        loss_function = criterion(out, label)
        losses.append(loss_function.item())
        loss_function.backward()
        
        optimizer.step()
        
        accurate_labels += torch.sum(torch.argmax(out, dim=1) == label).cpu()
        all_labels += len(label)
            
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data0), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss_function.item(),
                (100. * accurate_labels / all_labels)))
        torch.save(model.state_dict(), 'siamese_net_crossEntropy_withDropout.pt')
    train_loss = np.mean(losses)
    train_accuracy = 100. * accurate_labels / all_labels
    print('\nTrain set: Average loss = {:.4f}, Train Accuracy = {:.4f}\n'.format(train_loss, train_accuracy))
   
    return train_loss, train_accuracy

def main(EPOCHS=1):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_test_accuracy = 0.0

    for epoch in range(0, EPOCHS):
        print("Epoch number: ", epoch)
        start_time = time.time()
        print("\nTraining:")
        train_loss, train_accuracy = train(net, device, train_dataloader, epoch)
        print("\nTesting:")
        test_loss, test_accuracy = test(net, device, test_dataloader)
        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))
        train_accuracies.append((epoch, train_accuracy))
        test_accuracies.append((epoch, test_accuracy))
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(net.state_dict(), 'siamese_net_crossEntropy_withDropout.pt')
        end_time = time.time()
        print("Time taken for running epoch {} is {:.3f} seconds.\n\n".format(epoch, end_time-start_time))


if __name__ == '__main__':
    main()