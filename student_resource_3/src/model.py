import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        ## input is 3 output is 32 channel and so on..
        
        
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv4=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1=nn.Linear(256*8*8*1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,10)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        x=self.pool(F.relu(self.conv4(x)))
        
        x=x.view(-1,256 * 8 * 8)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x