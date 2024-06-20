import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def train_client(args,train_data,net):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=args.momentum)
    
    epoch_loss = []
    for epoch in range(args.epochs):
        
        batch_loss = []

        for i,data in enumerate(train_data,0):
            inputs,labels, _ = data
            if torch.cuda.is_available():
                inputs, labels=inputs.cuda(), labels.cuda()
            # print(inputs.shape,"Inputs size")
            # print(labels.shape,"Labels size")
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs.shape)
            # labels=torch.squeeze(labels,1)
            # print(labels.shape)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(),sum(epoch_loss)/len(epoch_loss)

def finetune_client(args,train_data,net):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=args.momentum)
    
    epoch_loss = []
    for epoch in range(1):
        
        batch_loss = []

        for i,data in enumerate(train_data,0):
            inputs,labels, _ = data
            if torch.cuda.is_available():
                inputs, labels=inputs.cuda(), labels.cuda()
            # print(inputs.shape,"Inputs size")
            # print(labels.shape,"Labels size")
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs.shape)
            # labels=torch.squeeze(labels,1)
            # print(labels.shape)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(),sum(epoch_loss)/len(epoch_loss)


def test_client(args,test_data,net):
    correct = 0
    total = 0
    test_loss = 0
    accuracy = 0
    correct_weight=0
    accuracy_weight=0

    with torch.no_grad():
        for idx, (data, target, data_weight) in enumerate(test_data):
            if torch.cuda.is_available():
                data, target, data_weight = data.cuda(), target.cuda(), data_weight.cuda()
            log_probs = net(data)
            # target=torch.squeeze(target,1)
            # print(target.shape,"Target")
            # print(data_weight.shape,"Data Weight")
            # print(log_probs.shape,"log_probs")
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            # print(y_pred.shape)
            equal_tensor=y_pred.eq(target.data.view_as(y_pred)).float()
            # print(equal_tensor)
            # print(data_weight)
            correct_weight += torch.mul(equal_tensor,data_weight).cpu().sum()
            # print(equal_tensor.shape)
            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()
        test_loss /= len(test_data.dataset)
        accuracy = 100.00 * correct / len(test_data.dataset)
        accuracy_weight = 100.00 * correct_weight / len(test_data.dataset)

    return accuracy,test_loss,accuracy_weight
