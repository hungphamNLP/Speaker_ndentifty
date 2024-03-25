import numpy as np
import torch
import tqdm
import time
from model import *
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


if __name__ == '__main__':
    test(net,device,test_dataloader)