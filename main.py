import argparse
import sys
import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import logging
import random
import torch.distributed as dist
import torchvision.models as models

# Parsing inputs - Using argparse
input_rank = -1
input_master_ip = ''
input_num_nodes = -1

parser = argparse.ArgumentParser(description='P2B: Distributed Data Parallel Training : Using AllReduce')
parser.add_argument('--rank', metavar='0', type=int, help='an integer for the accumulator', required=True)
parser.add_argument('--master-ip', metavar='tcp://10.10.1.1:6585', type=str, help='an integer for the accumulator', required=True)
parser.add_argument('--num-nodes', metavar='4', type=int, help='an integer for the accumulator', required=True)
parser.add_argument('--epochs', metavar='1', type=int, help='number of training epochs', required=True)
parser.add_argument('--freezelayer', metavar='1', type=int, help='number of layers to be frozen', required=True)
parser.add_argument('--gradnormfreeze', metavar='0', type=int, help='number of layers to be frozen', required=True)

args = parser.parse_args()
input_rank = int(args.rank)
input_master_ip = args.master_ip
input_num_nodes = int(args.num_nodes)
epochs = int(args.epochs) # total number of epochs
freezelayer = int(args.freezelayer) # total number of epochs
grad_norm_freeze = int(args.gradnormfreeze) # 0(no) or 1(yes) depending on grad norm freeze value

device = "cpu"
torch.set_num_threads(input_num_nodes)
torch.manual_seed(0)
np.random.seed(0)

torch.distributed.init_process_group('gloo', init_method=input_master_ip, rank=input_rank, world_size=input_num_nodes)

batch_size = 64 # batch for one node

model = models.resnet18(pretrained=True)

def update_gradients():
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.requires_grad is True:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def freeze():
    i=0
    for child in model.children():
        i+=1
        if i>=freezelayer:
            continue
        for param in child.parameters():
            param.requires_grad = False

def freeze_grad(n):
    i=0
    for child in model.children():
        i+=1
        if i>=n:
            continue
        for param in child.parameters():
            param.requires_grad = False
            
def check_norm_and_freeze(epoch,prev_model=None):
    if epoch<=0 or grad_norm_freeze==0 or prev_model==None:
        return
  
    i=0
    diff = []
    # calculate change in gradient for each block
    for child1,child2 in zip(model.children(),prev_model.children()):
        i+=1
        running_diff_sum = 0
        s = 0
        for param1,param2 in zip(child1.parameters(),child2.parameters()):
            if param1.grad is not None and param2.grad is not None:
                #print('i: ' + str(i))
                #print(param.grad.shape)
                p = param1.grad - param2.grad
                running_diff_sum += p.norm(2)
                s += param2.grad.norm(2)
        #running_diff_sum /= s
        diff.append(running_diff_sum)
    
    # find the block with lowest change and freeze till that block
    index = diff.index(min(diff))
    print('epoch : ' + str(epoch))
    print('freezing : ' + str(index+1))
    freeze_grad(index+1)
        

def train_model(train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    # remember to exit the train loop at end of the epoch
    running_loss = 0.0
    freeze()

    forwardT = 0.0
    backwardT = 0.0
    optimizerT = 0.0


    for batch_idx, (data, target) in enumerate(train_loader):
        
        start = time.time()

        optimizer.zero_grad()

        # forward + backward + optimize
        startF = time.time()
        outputs = model(data)
        loss = criterion(outputs, target)
        endF = time.time()
        forwardT += round(endF-startF,2)

        startB = time.time()
        loss.backward()
        update_gradients()
        endB = time.time()
        backwardT += round(endB-startB,2)

        startO = time.time()
        optimizer.step()
        endO = time.time()
        optimizerT += round(endO-startO,2)

        end = time.time()
        elapsed = str(round(end-start,2))

        # if batch_idx >=1 and batch_idx<=9:
        #     print('Batch: ' + str(batch_idx+1), 'Elapsed time: ' + elapsed + 's')

        # print statistics
        running_loss += loss.item()
        if batch_idx % 20 == 19:    # print every 20th mini-batch
            # print('[Epoch: %d, Batch: %5d] Loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 20))
            running_loss = 0.0

    print('[Epoch: %d] Forward: %.3f, Backward: %.3f, Optimizer: %.3f' % (epoch + 1, forwardT, backwardT, optimizerT))

def test_model(test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set, num_replicas=input_num_nodes, rank=input_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=input_num_nodes,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=(train_sampler is None),
                                                    pin_memory=True)

    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=input_num_nodes,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    # model = mdl.VGG11()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    
    prev_model = None
    # running training for one epoch
    for epoch in range(epochs):
        train_model(train_loader, optimizer, training_criterion, epoch)
        # after gradients have been fetched, check for grad norm
        check_norm_and_freeze(epoch, prev_model)
        prev_model = copy.deepcopy(model)
        test_model(test_loader, training_criterion)

if __name__ == "__main__":
    main()
