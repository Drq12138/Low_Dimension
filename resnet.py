'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
from timeit import repeat
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
# from replace_resnet import reResNet

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

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


# class new_bn(nn.Module):
#     def __init__(self, feature, weight, bias, new_param):
#         super(new_bn, self).__init__()
#         self.feature = feature              
#         self.weight = weight                # [low, feature]
#         self.bias = bias                    # [low, feature]
#         self.new_param = new_param          # [low]
    
#     def forward(self,x):
#         for i in range(40):
#             if i == 0:
#                 out = 


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class newBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, new_param, P_weight,layer_index,conv_index,conv_shape,origin_weight, stride=1, option='A', low_dimension = 40):
        super(newBasicBlock, self).__init__()
        # self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride),padding=(0,1,1), bias=False)
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

    def forward(self, x):
        out = torch.nn.functional.conv2d(x, self.origin_weight[self.conv_index[self.layer_index[0]][0]:self.conv_index[self.layer_index[0]][1]].reshape(self.conv_shape[self.layer_index[0]]), stride=self.stride, padding=1)


        for i in range(40):
            # if i == 0:
            #     out = self.new_param[i] * torch.nn.functional.conv2d(x, self.P_weight[i,self.conv_index[self.layer_index[0]][0]:self.conv_index[self.layer_index[0]][1]].reshape(self.conv_shape[self.layer_index[0]]), stride=self.stride, padding=1)
            # else:
            out += self.new_param[i] * torch.nn.functional.conv2d(x, self.P_weight[i,self.conv_index[self.layer_index[0]][0]:self.conv_index[self.layer_index[0]][1]].reshape(self.conv_shape[self.layer_index[0]]), stride=self.stride, padding=1)
        
        out = F.relu(self.bn1(out))
        out2 = self.new_param[i] * torch.nn.functional.conv2d(out, self.origin_weight[self.conv_index[self.layer_index[1]][0]:self.conv_index[self.layer_index[1]][1]].reshape(self.conv_shape[self.layer_index[1]]), stride=1, padding=1)

        for i in range(40):
            # if i == 0:
            #     out2 = self.new_param[i] * torch.nn.functional.conv2d(out, self.P_weight[i,self.conv_index[self.layer_index[1]][0]:self.conv_index[self.layer_index[1]][1]].reshape(self.conv_shape[self.layer_index[1]]), stride=1, padding=1)
            # else:
            out2 += self.new_param[i] * torch.nn.functional.conv2d(out, self.P_weight[i,self.conv_index[self.layer_index[1]][0]:self.conv_index[self.layer_index[1]][1]].reshape(self.conv_shape[self.layer_index[1]]), stride=1, padding=1)
        
        out2 = self.bn2(out2)
        out2 += self.shortcut(x)
        out2 = F.relu(out2)
        return out2


        # out = x.unsqueeze(2).repeat(1,1,self.lowd,1,1)
        # out = self.conv1(out)
        # new_out1 = self.new_param[None, None, :, None, None].expand_as(out)
        # out = out * new_out1
        # out = out.mean(2)
        # out = F.relu(self.bn1(out))

        # out = out.unsqueeze(2).repeat(1,1,self.lowd,1,1)
        # out = self.conv2(out)
        # new_out2 = self.new_param[None, None, :, None, None].expand_as(out)
        # out = out * new_out2
        # out = out.mean(2)
        # out = self.bn2(out)

        # out += self.shortcut(x)
        # out = F.relu(out)
        # return out

    # def __init__(self, in_planes, planes, stride=1, option='A'):
    #     super(BasicBlock, self).__init__()
    #     self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    #     self.bn1 = nn.BatchNorm2d(planes)
    #     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    #     self.bn2 = nn.BatchNorm2d(planes)

    #     self.shortcut = nn.Sequential()
    #     if stride != 1 or in_planes != planes:
    #         if option == 'A':
    #             """
    #             For CIFAR10 ResNet paper uses option A.
    #             """
    #             self.shortcut = LambdaLayer(lambda x:
    #                                         F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
    #         elif option == 'B':
    #             self.shortcut = nn.Sequential(
    #                  nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
    #                  nn.BatchNorm2d(self.expansion * planes)
    #             )

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.bn2(self.conv2(out))
    #     out += self.shortcut(x)
    #     out = F.relu(out)
    #     return out



class testResNet(nn.Module):
    def __init__(self, block, num_blocks,P, conv_names, conv_index, conv_shapes, origin_weight, low_d = 40):
        super(testResNet,self).__init__()
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


        self.bn1 = nn.BatchNorm2d(16)
        # layer1_conv_index = [i for i in range(1,7)]
        # layer2_conv_index = [i for i in range(7,13)]
        # layer3_conv_index = [i for i in range(13,19)]
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
        out = torch.nn.functional.conv2d(x, self.origin_weight[self.conv_index[0][0]:self.conv_index[0][1]].reshape(self.conv_shape[0]), stride=1, padding=1)
        for i in range(40):
            # if i == 0:
            #     out = self.new_param[i] * torch.nn.functional.conv2d(x, self.P_weight[i,self.conv_index[0][0]:self.conv_index[0][1]].reshape(self.conv_shape[0]), stride=1, padding=1)
            # else:
            out += self.new_param[i] * torch.nn.functional.conv2d(x, self.P_weight[i,self.conv_index[0][0]:self.conv_index[0][1]].reshape(self.conv_shape[0]), stride=1, padding=1)
        
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class newResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, low_dimension = 40):
        super(newResNet, self).__init__()
        self.in_planes = 16
        self.lowd = low_dimension
        self.new_param = nn.Parameter(torch.rand(self.lowd))
        self.register_parameter("Ablah",self.new_param)

        self.mut_conv1 = nn.Conv3d(3,16,(self.lowd,3,3), stride=(1,1,1),padding=(0,1,1), bias=False)


        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, [i for i in range(1,7)])
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, [i for i in range(7,13)])
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, [i for i in range(13,19)])
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, layer_index):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.new_param ,stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
        # return layers

    def forward(self, x):
        x = x.unsqueeze(2).repeat(1,1,self.lowd,1,1)
        out = self.mut_conv1(x)
        new_out = self.new_param[None, None, :, None, None].expand_as(out)
        out = out * new_out
        out = out.mean(2)

        out = F.relu(self.bn1(out))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


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




def resnet8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)

def replace_resnet20(P, conv_names, conv_index, conv_shapes, origin_weight):
    # return newResNet(BasicBlock, [3, 3, 3], num_classes)
    # return testResNet(newBasicBlock, [3, 3, 3], P, conv_names, conv_index, conv_shapes, origin_weight)
    return reResNet(reBlock, [3, 3, 3], P, conv_names, conv_index, conv_shapes, origin_weight)

def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes)


def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
