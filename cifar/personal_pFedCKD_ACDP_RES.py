
import matplotlib

from utility.paint import paintpathological, paintdirichlet

import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torchsummary import summary
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

# Directory where the json file of arguments will be present
directory = './Parse_Files'

# Import different files required for loading dataset, model, testing, training
from utility.LoadSplit import Load_Dataset, Load_Model
# from utility.options import args_parser
from models.Update_SKD_2 import train_client, test_client, _client_sampling, clip_and_add_noise_our, subtract, add, \
    trainKD_client, get_sigma_or_epsilon, selFD_train
from models.Fed import FedAvg
from models.test import test_img

torch.manual_seed(0)

if __name__ == '__main__':

    # Initialize argument dictionary
    args = {}

    # From Parse_Files folder, get the name of required parse file which is provided while running this script from bash
    f = directory + '/' + str(sys.argv[1])
    print(f)

    with open(f) as json_file:
        args = json.load(json_file)

    # Taking hash of config values and using it as filename for storing model parameters and logs
    param_str = json.dumps(args)
    file_name = md5(param_str.encode()).hexdigest()

    # Converting args to easydict to access elements as args.device rather than args[device]
    args = easydict.EasyDict(args)
    print(args)
    current_time = time.time()
    # current_datetime = datetime.now()
    selected_params = ["partition", 'local_ep', 'overlapping_classes', 'model', 'dataset', "local_bs",
                       "dp_clip", "lr", "epochs", "split_ratio", "num_users", "temperature"]
    name = 'SelKD_'
    for param in selected_params:
        param_value = args.get(param)
        name += f'{param}_{param_value}_'
    name = name[:-1]
    # 可能是提前出触发了日志文件
    file_name = name
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)



    # Save configurations by making a file using hash value
    with open('./config/parser_{}.txt'.format(file_name), 'w') as outfile:
        json.dump(args, outfile, indent=4)


    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(current_time))
    SUMMARY = os.path.join('./results', f"{file_name}_{timestamp}")  # 将多个字符串组合程一个完整的路径
    args.summary = SUMMARY
    os.makedirs(SUMMARY)  # 创建一个文件夹SUMMARY

    # Setting the device - GPU or CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # net_glob = Load_Model(args=args)
    # print(net_glob)
    # exit()
    # Load the training and testing datasets

    dataset_train, dataset_test, dict_users = Load_Dataset(args=args)

    # if args.partition == "n_cls":
    #     paintpathological(args, dataset_train, dict_users)
    # if args.partition == "dir":
    #     paintdirichlet(args,dataset_train,dict_users)
    # Initialize Global Server Model
    net_glob = Load_Model(args=args)
    # print(net_glob)
    net_glob.train()

    # Print name of the architecture - 'MobileNet or ResNet or NewNet'
    print(args.model)

    # copy weights
    w_glob = net_glob.state_dict()

    # Set up log file
    logging.basicConfig(filename='./log/FedSKD/{}.txt'.format(file_name), format='%(message)s', level=logging.DEBUG)
    logging.info(args)
    tree = lambda: defaultdict(tree)
    stats = tree()
    writer = SummaryWriter(args.summary)

    # splitting user data into training and testing parts
    train_data_users = {}
    test_data_users = {}
    if args.partition == "n_cls":
        for i in range(args.num_users):
            # dict_users[i] = list(dict_users[i])
            train_data_users[i] = list(random.sample(dict_users[i], int(args.split_ratio * len(dict_users[i]))))
            test_data_users[i] = list(set(dict_users[i]) - set(train_data_users[i]))
    if args.partition == "dir":
        for i in range(args.num_users):
            dict_users[i] = list(dict_users[i])
            train_data_users[i] = list(random.sample(dict_users[i], int(args.split_ratio * len(dict_users[i]))))
            test_data_users[i] = list(set(dict_users[i]) - set(train_data_users[i]))

    # exit()
    # local models for each client
    local_nets = {}
    for i in range(0, args.num_users):
        local_nets[i] = Load_Model(args=args)
        local_nets[i].train()
        local_nets[i].load_state_dict(w_glob)


    # Start training

    logging.info("Training")

    start = time.time()
    w_locals = {}
    best_test_accuracy = 0.0
    if args.DP:
        epsilon, noise_multiplier = get_sigma_or_epsilon(round=args.epochs, args=args)
        logging.info("Statistical last privacy budget {} :".format(epsilon))
        logging.info("Statistical last noise_multiplier {} :".format(noise_multiplier))
        w_glob_new = copy.deepcopy(w_glob)
        for iter in range(args.epochs):
            last_w_global = copy.deepcopy(w_glob_new)
            print('Round {}'.format(iter))

            logging.info("---------Round {}---------".format(iter))

            w_locals_agg, loss_locals = [], []

            client_indexes = _client_sampling(iter, args)
            # 将选择的客户端进行排序
            client_indexes = np.sort(client_indexes)

            for i in client_indexes:
                local_nets[i] = Load_Model(args=args)
                local_nets[i].train()
                # w_locals[i] = copy.deepcopy(w_glob)
                if iter == 0:
                    local_nets[i].load_state_dict(w_glob)
                else:
                    if i in w_locals:
                        for j in list(w_glob_new.keys())[0:args.base_layers]:
                            w_locals[i][j] = copy.deepcopy(w_glob_new[j])
                        local_nets[i].load_state_dict(w_locals[i], False)
                    else:
                        local_nets[i].load_state_dict(w_glob_new, False)

            for idx in client_indexes:
                w, loss = selFD_train(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                w_nabala = copy.deepcopy(w)

                # nabala = copy.deepcopy(subtract(w_nabala, w_glob_new))
                if iter == 0:
                    nabala = copy.deepcopy(subtract(w_nabala, w_glob))
                else:
                    nabala = copy.deepcopy(subtract(w_nabala, w_glob_new))

                nabala = clip_and_add_noise_our(nabala, args)
                selected_nabala_keys = list(nabala.keys())[0:args.base_layers]
                selected_nabala_values = list(nabala.values())[0:args.base_layers]
                nabala_w_base = {}
                for k, v in zip(selected_nabala_keys, selected_nabala_values):
                    nabala_w_base[k] = v
                w_locals[idx] = w
                w_locals_agg.append(nabala_w_base)
                loss_locals.append(copy.deepcopy(loss))

            # store testing and training accuracies of the model before global aggregation
            s = 0
            k = 0
            for cur_clnt in client_indexes:
                # logging.info("Client {}:".format(i))
                acc_train, loss_train = test_client(args, dataset_train, train_data_users[cur_clnt],
                                                    local_nets[cur_clnt])
                acc_test, loss_test = test_client(args, dataset_train, test_data_users[cur_clnt], local_nets[cur_clnt])
                stats[cur_clnt][iter]['Before Training accuracy'] = acc_train
                stats[cur_clnt][iter]['Before Test accuracy'] = acc_test
                writer.add_scalar(str(cur_clnt) + '/Before Training accuracy', acc_train, iter)
                writer.add_scalar(str(cur_clnt) + '/Before Test accuracy', acc_test, iter)
                k += acc_train
                s += acc_test

            k /= (args.num_users * args.frac)
            s /= (args.num_users * args.frac)
            if s > best_test_accuracy:
                best_test_accuracy = s

            loss_avg = sum(loss_locals) / len(loss_locals)
            logging.info("Average Client accuracy on their train data: {: .3f}".format(k))
            logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
            logging.info("Best accuracy on their test data: {: .3f}".format(best_test_accuracy))
            logging.info("Average loss of clients: {: .3f}".format(loss_avg))
            stats['Before Average'][iter] = s
            writer.add_scalar('Average' + '/Before train accuracy', k, iter)
            writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
            writer.add_scalar('Average' + '/loss', loss_avg, iter)

            # hyperparameter = number of layers we want to keep in the base part
            base_layers = args.base_layers
            w_glob_avg = FedAvg(w_locals_agg)
            w_glob_new = add(last_w_global, w_glob_avg)
            # update global weights
            # w_glob_new = FedAvg(w_locals_agg)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob_new, False)
    else:
        for iter in range(args.epochs):

            print('Round {}'.format(iter))

            logging.info("---------Round {}---------".format(iter))

            w_locals_agg, loss_locals = [], []

            client_indexes = _client_sampling(iter, args)
            # 将选择的客户端进行排序
            client_indexes = np.sort(client_indexes)

            for i in client_indexes:
                local_nets[i] = Load_Model(args=args)
                local_nets[i].train()
                # w_locals[i] = copy.deepcopy(w_glob)
                if iter == 0:
                    local_nets[i].load_state_dict(w_glob)
                else:
                    if i in w_locals:
                        for j in list(w_glob_new.keys())[0:args.base_layers]:
                            w_locals[i][j] = copy.deepcopy(w_glob_new[j])
                        local_nets[i].load_state_dict(w_locals[i], False)
                    else:
                        local_nets[i].load_state_dict(w_glob_new, False)

            for idx in client_indexes:
                w, loss = selFD_train(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                selected_w_keys = list(w.keys())[:args.base_layers]
                selected_w_values = list(w.values())[:args.base_layers]
                nabala_w_base = {}
                for k, v in zip(selected_w_keys, selected_w_values):
                    nabala_w_base[k] = v

                w_locals[idx] = w
                w_locals_agg.append(nabala_w_base)
                loss_locals.append(copy.deepcopy(loss))

            # store testing and training accuracies of the model before global aggregation
            s = 0
            k = 0
            for cur_clnt in client_indexes:
                # logging.info("Client {}:".format(i))
                acc_train, loss_train = test_client(args, dataset_train, train_data_users[cur_clnt], local_nets[cur_clnt])
                acc_test, loss_test = test_client(args, dataset_train, test_data_users[cur_clnt], local_nets[cur_clnt])
                stats[cur_clnt][iter]['Before Training accuracy'] = acc_train
                stats[cur_clnt][iter]['Before Test accuracy'] = acc_test
                writer.add_scalar(str(cur_clnt) + '/Before Training accuracy', acc_train, iter)
                writer.add_scalar(str(cur_clnt) + '/Before Test accuracy', acc_test, iter)
                k += acc_train
                s += acc_test

            k /= (args.num_users * args.frac)
            s /= (args.num_users * args.frac)
            if s > best_test_accuracy:
                best_test_accuracy = s

            loss_avg = sum(loss_locals) / len(loss_locals)
            logging.info("Average Client accuracy on their train data: {: .3f}".format(k))
            logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
            logging.info("Best accuracy on their test data: {: .3f}".format(best_test_accuracy))
            logging.info("Average loss of clients: {: .3f}".format(loss_avg))
            stats['Before Average'][iter] = s
            writer.add_scalar('Average' + '/Before train accuracy', k, iter)
            writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
            writer.add_scalar('Average' + '/loss', loss_avg, iter)

            # hyperparameter = number of layers we want to keep in the base part
            base_layers = args.base_layers

            # update global weights
            w_glob_new = FedAvg(w_locals_agg)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob_new, False)

    end = time.time()

    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    logging.info(
        "Global Server Model - Train Acc: {:.3f}, Train Loss: {:.3f}, Test Acc: {:.3f}, Test Loss: {:.3f}".format(
            acc_train, loss_train, acc_test, loss_test))

    logging.info("Training Time: {}s".format(end - start))
    logging.info("End of Training")

    # save model parameters
    torch.save(net_glob.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    for i in range(args.num_users):
        torch.save(local_nets[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))

    # test global model on training set and testing set

    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")

    # testing global model on individual client's test data
    # s = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))

    dill.dump(stats, open(os.path.join(args.summary, 'stats.pkl'), 'wb'))
    writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])