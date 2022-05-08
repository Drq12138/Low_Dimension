from turtle import forward
from matplotlib.pyplot import cla
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BasicNet(torch.nn.Module):
    def __init__(self):
        super(BasicNet,self).__init__()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

        self.conv1 = nn.Conv1d(1,10,3)
        self.conv2 = nn.Conv1d(10,16,3)
        self.conv3 = nn.Conv1d(16,10,5)
        self.optimizer = torch.optim.Adam(self.parameters())

        self.layer1 = nn.Sequential(nn.Conv2d(1,25,kernel_size=3), nn.BatchNorm2d(25), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(25, 50, kernel_size=3), nn.BatchNorm2d(50), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(nn.Linear( 50*5*5, 1024),nn.ReLU(inplace=True),nn.Linear(1024, 128),nn.ReLU(inplace=True),nn.Linear(128, 10))
    
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = y.view(y.size(0), -1)
        y = self.layer5(y)
        
        return y

batch_s = 200
epochs = 100
   



if __name__=="__main__":
    data_path = './../net_data/'
    epochs = 100
    trainDataset = torchvision.datasets.MNIST(root=data_path,train=True,transform=transforms.ToTensor(), download=True)
    testDataset = torchvision.datasets.MNIST(root=data_path,train=False,transform=transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(dataset=trainDataset,batch_size=batch_s,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testDataset,batch_size=batch_s,shuffle=True)
    
    origin_model = BasicNet()
    origin_model.train()
    origin_model.to(device)
    
    for 
    
    
    