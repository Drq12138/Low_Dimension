
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class reBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, new_param, P_weight,layer_index,conv_index,conv_shape,origin_weight, stride=1, option='A', low_dimension = 40):
        super(reBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(planes)
        self.new_param = new_param
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,1,1),padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lowd = low_dimension
        self.conv_index = conv_index
        self.P_weight = P_weight
        self.layer_index = layer_index
        self.stride = stride
        self.conv_shape = conv_shape
        self.origin_weight = origin_weight
        

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.conv1_weight = self.origin_weight[self.conv_index[self.layer_index[0]][0]:self.conv_index[self.layer_index[0]][1]].reshape(self.conv_shape[self.layer_index[0]])

        for i in range(40):
            self.conv1_weight += self.new_param[i] * self.P_weight[i,self.conv_index[self.layer_index[0]][0]:self.conv_index[self.layer_index[0]][1]].reshape(self.conv_shape[self.layer_index[0]])
        
        self.conv2_weight = self.origin_weight[self.conv_index[self.layer_index[1]][0]:self.conv_index[self.layer_index[1]][1]].reshape(self.conv_shape[self.layer_index[1]])

        for i in range(40):
            self.conv2_weight += self.new_param[i] * self.P_weight[i,self.conv_index[self.layer_index[1]][0]:self.conv_index[self.layer_index[1]][1]].reshape(self.conv_shape[self.layer_index[1]])
    
    
    def forward(self, x):
        out = torch.nn.functional.conv2d(x, self.conv1_weight, stride=self.stride, padding=1)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class reResNet(nn.Module):
    def __init__(self, block, num_blocks, P, conv_names, conv_index, conv_shapes, origin_weight, low_d = 40):
        super(reResNet,self).__init__()
        # self.layer1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  #kernel size(16, 3, 3,3)
        self.conv_names = conv_names
        self.conv_index = conv_index
        self.conv_shape = conv_shapes
        self.origin_weight = origin_weight          
        self.in_planes = 16
        self.lowd = low_d
        self.new_param = nn.Parameter(torch.rand(self.lowd))
        self.register_parameter("Ablah",self.new_param)
        self.P_weight = P                   # P: [40, 269722]/[n_low_dimension, n_parameter]
        

        self.conv1_weight = self.origin_weight[self.conv_index[0][0]:self.conv_index[0][1]].reshape(self.conv_shape[0])

        for i in range(self.lowd):
            self.conv1_weight += self.new_param[i] * self.P_weight[i,self.conv_index[0][0]:self.conv_index[0][1]].reshape(self.conv_shape[0])
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], [[1, 2], [3, 4], [5, 6]], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], [[7, 8], [9, 10], [11, 12]], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], [[13, 14], [15, 16],[17, 18]], stride=2)
        self.linear = nn.Linear(64, 10)

        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, layer_index, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, self.new_param, self.P_weight, layer_index[i], self.conv_index, self.conv_shape,self.origin_weight, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)




    def forward(self,x):
        out = torch.nn.functional.conv2d(x, self.conv1_weight, stride=1, padding=1)
        out = F.relu(self.bn1(out))



def replace_resnet20(P, conv_names, conv_index, conv_shapes, origin_weight):
    # return newResNet(BasicBlock, [3, 3, 3], num_classes)
    return reResNet(reBlock, [3, 3, 3], P, conv_names, conv_index, conv_shapes, origin_weight)