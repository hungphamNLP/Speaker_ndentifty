import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*227*227, 512),
            nn.ReLU(inplace=True),
            ## Using 30% dropout probability
            nn.Dropout(p=0.3),
            )
        ## Final linear layer output dim is 2: where one of them
        ## corresponds to saying the datapoints match and the other indicates that it doesn't
        self.fc2 = nn.Linear(512, 2)

    def forward_per_datapoint(self, x):
        print(x.shape)
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_per_datapoint(input1)
        output2 = self.forward_per_datapoint(input2)
        return self.fc2(torch.abs(output1 - output2))
    
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, label):
        label = label.long()
        loss = F.cross_entropy(output, label)
        return loss
    

counter = []
loss_history = [] 
iteration_number= 0

net = SiameseNetwork().to(device)
criterion = CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr = 0.001)