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
from load_model import Load_Model
# from model import MobileNet, ResNet
from update import train_client, test_client, finetune_client
import load_data
from fed import Fedavg

import copy
import time
import random
import logging
import json
from hashlib import md5
import copy
import easydict
import os
import sys
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import pickle
import dill

directory = './Parse'

if __name__ == '__main__':
    
    # Arguments
    args = {}
    
    # Config filename
    f = directory+'/'+str(sys.argv[1])
    print(f)
    
    with open(f) as json_file:  
        args = json.load(json_file)

    # Taking hash of config values and using it as filename for storing model parameters and logs
    param_str = json.dumps(args)
    file_name = md5(param_str.encode()).hexdigest()

    # Converting args to easydict to access elements as args.device
    args = easydict.EasyDict(args)
    print(args)



    SUMMARY = os.path.join('./results',file_name)
    args.summary=SUMMARY
    os.makedirs(SUMMARY)

    # Save configurations 
    with open('./config/{}.txt'.format(file_name),'w') as outfile:
        json.dump(args,outfile,indent=4)

    # Setting the device
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Set up log file
    logging.basicConfig(filename='./log/{}.log'.format(file_name),format='%(message)s',level=logging.DEBUG)
   

    tree=lambda : defaultdict(tree)
    stats=tree()
    writer = SummaryWriter(args.summary)
    
    #initialize global model
    # net_glob = MobileNet(args=args)
    net_glob = Load_Model(args=args)
    # print(net_glob)
    if torch.cuda.is_available():
        net_glob = net_glob.cuda()
    net_glob.train()
    w_glob = net_glob.state_dict()

    # load data for each client
    dirt = './users_data'
    train_data_users, test_data_users, train_data_size = {},{},{}

    for i in os.listdir(dirt):
        if (i[0]=='.'):
            continue
        j = str(i)
        train_data_size[j],train_data_users[j],test_data_users[j] = load_data.new_load_dataset(j,args=args)

        
    #initialize the local models
    local_nets = {}
    for i in train_data_users.keys():
        # local_nets[i] = MobileNet(args=args)
        local_nets[i] = Load_Model(args=args)
        if torch.cuda.is_available(): 
            local_nets[i] = local_nets[i].cuda()
        local_nets[i].train()
        local_nets[i].load_state_dict(w_glob)

        
    #store number of training examples of each client
    count,users = [],[]
    for i in train_data_size:
        count.append(train_data_size[i])
        users.append(i)

    
    logging.info("Training")
    # Start training
    start = time.time()
    
    # acc_test, loss_test = test_client(args,test_data_users[users[0]],local_nets[users[0]])
    # exit()

    for j in range(args.rounds):
        print('Round {}'.format(j))
        logging.info("---------Round {}---------".format(j))

        w_locals,loss_locals = [], []

        for i in local_nets.keys():

            print('User {}'.format(i))
            w,loss = train_client(args,train_data_users[i],local_nets[i])
            w_locals.append(w)
            loss_locals.append(loss)

        base_layers = args.base_layers

        logging.info("Testing Client Models before aggregation")
        logging.info("")
        
        s = 0
        s1 = 0
        for i in range(len(users)):
            logging.info("Client {}:".format(i))
            acc_train, loss_train, acc_train_weight = test_client(args,train_data_users[users[i]],local_nets[users[i]])
            acc_test, loss_test, acc_test_weight = test_client(args,test_data_users[users[i]],local_nets[users[i]])
            logging.info("Training accuracy: {:.3f}".format(acc_train))
            logging.info("Testing accuracy: {:.3f}".format(acc_test))
            logging.info("Training accuracy weight: {:.3f}".format(acc_train_weight))
            logging.info("Testing accuracy weight: {:.3f}".format(acc_test_weight))
            logging.info("")
            
            stats[users[i]][j]['Before Training accuracy']=acc_train
            stats[users[i]][j]['Before Test accuracy']=acc_test
            stats[users[i]][j]['Before Training accuracy weight']=acc_train_weight
            stats[users[i]][j]['Before Test accuracy weight']=acc_test_weight

            writer.add_scalar(users[i]+'/Before Training accuracy',acc_train,j)
            writer.add_scalar(users[i]+'/Before Test accuracy',acc_test,j)
            writer.add_scalar(users[i]+'/Before Training accuracy weight',acc_train_weight,j)
            writer.add_scalar(users[i]+'/Before Test accuracy weight',acc_test_weight,j)



            s += acc_test
            s1 += acc_test_weight
        s /= len(users)
        s1 /= len(users)
        logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
        logging.info("Average Client accuracy weight on their test data: {: .3f}".format(s1))
        
        w_glob = Fedavg(w_locals,count)

        net_glob.load_state_dict(w_glob)
        for idx in range(len(local_nets.keys())):
            for i  in list(w_glob.keys())[0:base_layers]:
                w_locals[idx][i] = copy.deepcopy(w_glob[i]) 

            local_nets[users[idx]].load_state_dict(w_locals[idx])

        logging.info("Testing Client Models after aggregation")
        logging.info("")
        s = 0
        s1 = 0
        for i in range(len(users)):
            logging.info("Client {}:".format(i))
            acc_train, loss_train, acc_train_weight = test_client(args,train_data_users[users[i]],local_nets[users[i]])
            acc_test, loss_test, acc_test_weight = test_client(args,test_data_users[users[i]],local_nets[users[i]])
            logging.info("Training accuracy: {:.3f}".format(acc_train))
            logging.info("Testing accuracy: {:.3f}".format(acc_test))
            logging.info("Training accuracy weight: {:.3f}".format(acc_train_weight))
            logging.info("Testing accuracy weight: {:.3f}".format(acc_test_weight))
            logging.info("")

            stats[users[i]][j]['After Training accuracy']=acc_train
            stats[users[i]][j]['After Test accuracy']=acc_test
            stats[users[i]][j]['After Training accuracy weight']=acc_train_weight
            stats[users[i]][j]['After Test accuracy weight']=acc_test_weight

            writer.add_scalar(users[i]+'/After Training accuracy',acc_train,j)
            writer.add_scalar(users[i]+'/After Test accuracy',acc_test,j)
            writer.add_scalar(users[i]+'/After Training accuracy weight',acc_train_weight,j)
            writer.add_scalar(users[i]+'/After Test accuracy weight',acc_test_weight,j)

            s += acc_test
            s1 += acc_test_weight
        s /= len(users)
        s1 /= len(users)
        logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
        logging.info("Average Client accuracy weight on their test data: {: .3f}".format(s1))


        loss_avg = sum(loss_locals) / len(loss_locals)
        logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
       
        ###FineTuning
        if args.finetune:
            # print("FineTuning")
            personal_params=list(w_glob.keys())[base_layers:]
            for idx in (local_nets.keys()):
                for i,param in enumerate(local_nets[idx].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad=False
                w,loss = finetune_client(args,train_data_users[idx],net = local_nets[idx])
                for i,param in enumerate(local_nets[idx].named_parameters()):
                    if param[0] not in personal_params:
                        param[1].requires_grad=True

            s = 0
            s1 = 0
            for i in range(len(users)):
                logging.info("Client {}:".format(i))
                acc_train, loss_train, acc_train_weight = test_client(args,train_data_users[users[i]],local_nets[users[i]])
                acc_test, loss_test, acc_test_weight = test_client(args,test_data_users[users[i]],local_nets[users[i]])
                logging.info("Training accuracy finetune: {:.3f}".format(acc_train))
                logging.info("Testing accuracy finetune: {:.3f}".format(acc_test))
                logging.info("Training accuracy finetune weight: {:.3f}".format(acc_train_weight))
                logging.info("Testing accuracy finetune weight: {:.3f}".format(acc_test_weight))
                logging.info("")

                stats[users[i]][j]['After finetune Training accuracy']=acc_train
                stats[users[i]][j]['After finetune Test accuracy']=acc_test
                stats[users[i]][j]['After finetune Training accuracy weight']=acc_train_weight
                stats[users[i]][j]['After finetune Test accuracy weight']=acc_test_weight

                writer.add_scalar(users[i]+'/After finetune Training accuracy',acc_train,j)
                writer.add_scalar(users[i]+'/After finetune Test accuracy',acc_test,j)
                writer.add_scalar(users[i]+'/After finetune Training accuracy weight',acc_train_weight,j)
                writer.add_scalar(users[i]+'/After finetune Test accuracy weight',acc_test_weight,j)

                s += acc_test
                s1 += acc_test_weight
            s /= len(users)
            s1 /= len(users)
            logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
            logging.info("Average Client accuracy weight on their test data: {: .3f}".format(s1))

            stats['After finetune Average'][iter]=s
            stats['After finetune weight Average'][iter]=s1

    end = time.time()

    logging.info("Training Time: {}s".format(end-start))
    logging.info("End of Training")

    
    torch.save(net_glob.state_dict(),'./state_dict/server_{}.pt'.format(file_name))
    for i in train_data_users:
        torch.save(local_nets[i].state_dict(),'./state_dict/client_{}_{}.pt'.format(i,file_name))



    dill.dump(stats,open(os.path.join(args.summary,'stats.pkl'),'wb'))
    writer.close()
    # print(stats['After finetune Average'], stats['After finetune weight Average'])
    
    for i in train_data_users:
        print(i)
        acc_train, loss_train, acc_train_weight = test_client(args,train_data_users[i],local_nets[i])
        acc_test, loss_test, acc_test_weight = test_client(args,test_data_users[i],local_nets[i])
        acc_test, loss_test, acc_test_weight = test_client(args,test_data_users[i],local_nets[i])
        print("Training accuracy: {:.3f}".format(acc_train))
        print("Training loss: {:.3f}".format(loss_train))
        print("Testing accuracy: {:.3f}".format(acc_test))
        print("Testing loss: {:.3f}".format(loss_test))
        print("Training accuracy weight: {:.3f}".format(acc_train_weight))
        print("Testing accuracy weight: {:.3f}".format(acc_test_weight))
