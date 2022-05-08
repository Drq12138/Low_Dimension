import argparse
import os
import time
import numpy as np
import random
from sklearn.decomposition import PCA
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torch
import torch.nn as nn
from utils import get_model, get_datasets, divide_param, build_new_model
# from replace_resnet import reResNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
train_acc, test_acc, train_loss, test_loss = [], [], [], []

class AverageMeter(object):
    # Computes and stores the average and current value

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    # Computes the precision@k for the specified values of k

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_model_param_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)

def set_seed(seed=233): 
    print ('Random Seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(train_loader, model, criterion, optimizer, epoch, args):
    global train_loss, train_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target_var = target.cuda()
        input_var = input.cuda()
        output = model(input_var)
        optimizer.zero_grad()
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader)-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    train_loss.append(losses.avg)
    train_acc.append(top1.avg)


def validate(val_loader, model, criterion, args):
    global test_acc, test_loss 
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # Compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # Measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    # Store the test loss and test accuracy
    test_loss.append(losses.avg)
    test_acc.append(top1.avg)

    return top1.avg




def main():
    parser = argparse.ArgumentParser(description='test the replace plan')
    parser.add_argument('--arch', '-a', default='resnet32')
    parser.add_argument('--datasets', default='CIFAR10')
    parser.add_argument('-j','--workers', default=4)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--start-epoch', default=0)
    parser.add_argument('--batch-size', default=128)
    parser.add_argument('--weight-decay', '--wd', default=1e-4)
    parser.add_argument('--print-freq', '-p', default=50)
    parser.add_argument('--save-dir', default='save_temp')
    parser.add_argument('--n_components', default=40)
    parser.add_argument('--params_start', default=0)
    parser.add_argument('--params_end', default=51)
    parser.add_argument('--randomseed', default=1)
    parser.add_argument('--corrupt', default=0, type=float,
                    metavar='c', help='noise level for training set')
    parser.add_argument('--smalldatasets', default=None, type=float, dest='smalldatasets', 
                    help='percent of small datasets')

    args = parser.parse_args()
    set_seed(args.randomseed)
    best_prec1 = 0
    P = None
    
    print ('we will save in: ' + args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = get_model(args)
    model.cuda()
    conv_names, conv_index, conv_shapes = divide_param(model)

    print ('params: from', args.params_start, 'to', args.params_end)
    W = []
    last_weight = None
    for i in range(args.params_start, args.params_end):
        model.load_state_dict(torch.load(os.path.join(args.save_dir,  str(i) +  '.pt')))
        temp_weight = get_model_param_vec(model)
        if last_weight is None:
            last_weight = temp_weight
        else:
            W.append(temp_weight-last_weight)
            last_weight = temp_weight
        
        #W.append(get_model_param_vec(model))
    W = np.array(W)
    print ('W:', W.shape)

    pca = PCA(n_components=args.n_components)
    pca.fit_transform(W)
    P = np.array(pca.components_)
    print ('ratio:', pca.explained_variance_ratio_)
    print ('P:', P.shape)

    P = torch.from_numpy(P).cuda()              # P: [40, 269722]/[n_low_dimension, n_parameter]

    cudnn.benchmark = True
    model.load_state_dict(torch.load(os.path.join(args.save_dir,  str(args.params_start) +  '.pt')))
    start_weight = get_model_param_vec(model)
    start_weight = torch.from_numpy(start_weight).cuda()

    new_model = build_new_model(P, conv_names, conv_index, conv_shapes, start_weight)
    # new_model = reResNet()
    # temp_weight = get_model_param_vec(new_model)
    new_model.cuda().to(device)
    new_conv_names, new_conv_index, new_conv_shapes = divide_param(new_model, True)

    train_loader, val_loader = get_datasets(args)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=1, momentum=0)
    end = time.time()
    end1 = end
    
    epoch_time = []
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)

        epoch_time.append(time.time() - end1)
        end1 = time.time()

        prec1 = validate(val_loader, model, criterion, args)

        best_prec1 = max(prec1, best_prec1)
    

    print ('total time:', time.time() - end)
    print ('train loss: ', train_loss)
    print ('train acc: ', train_acc)
    print ('test loss: ', test_loss)
    print ('test acc: ', test_acc)      
    print ('best_prec1:', best_prec1)
    print ('epoch time:', epoch_time)














if __name__=='__main__':
    main()