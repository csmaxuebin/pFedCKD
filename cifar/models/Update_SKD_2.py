import copy

import math
import torch
from torch import nn, autograd, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utility.sam import enable_running_stats, disable_running_stats

from utility.sam import SAM

from utility.differential_privacy import get_epsilon, get_noise_multiplier_from_epsilon


class DatasetSplit(Dataset):
    """
    Class DatasetSplit - To get datasamples corresponding to the indices of samples a particular client has from the actual complete dataset

    """

    def __init__(self, dataset, idxs):
        """

        Constructor Function

        Parameters:

            dataset: The complete dataset

            idxs : List of indices of complete dataset that is there in a particular client

        """
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        """

        returns length of local dataset

        """

        return len(self.idxs)

    def __getitem__(self, item):
        """
        Gets individual samples from complete dataset

        returns image and its label

        """
        image, label = self.dataset[self.idxs[item]]
        return image, label


# function to train a client
def CrossEntropy(args, outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs / args.temperature, dim=1)
    softmax_targets = F.softmax(targets / args.temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def selFD_train(args, dataset, train_idx, net):
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    sum_loss, total = 0.0, 0.0
    init = False
    # 计算预测值与实际标签之间的差异
    criterion = nn.CrossEntropyLoss()
    epoch_loss = []
    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):

            inputs, labels = images.to(args.device), labels.to(args.device)
            outputs, outputs_feature = net(inputs)
            ensemble = sum(outputs[:-1]) / len(outputs)
            ensemble.detach_()

            if init is False:
                #   init the adaptation layers.
                #   we add feature adaptation layers here to soften the influence from feature distillation loss
                #   the feature distillation in our conference version :  | f1-f2 | ^ 2
                #   the feature distillation in the final version : |Fully Connected Layer(f1) - f2 | ^ 2
                layer_list = []
                teacher_feature_size = outputs_feature[0].size(1)
                for index in range(1, len(outputs_feature)):
                    student_feature_size = outputs_feature[index].size(1)
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                net.adaptation_layers = nn.ModuleList(layer_list)
                net.adaptation_layers.cuda()
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
                #   define the optimizer here again so it will optimize the net.adaptation_layers
                init = True

            #   compute loss
            loss = torch.FloatTensor([0.]).to(args.device)

            #   for deepest classifier
            loss += criterion(outputs[0], labels)

            teacher_output = outputs[0].detach()
            teacher_feature = outputs_feature[0].detach()

            #   for shallow classifiers
            for index in range(1, len(outputs)):
                #   logits distillation
                loss += 0.5 * (CrossEntropy(args, outputs[index], teacher_output) + CrossEntropy(args, teacher_output, outputs[index])) * args.loss_coefficient
                loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)
                #   feature distillation

                if index != 1:
                    loss += torch.dist(net.adaptation_layers[index - 1](outputs_feature[index]), teacher_feature) * \
                            args.feature_loss_coefficient
                    #   the feature distillation loss will not be applied to the shallowest classifier 特征蒸馏损失不会应用于最浅的分类器
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# function to test a client
def _client_sampling(round_idx, args):
    client_num_per_round = int(args.frac * args.num_users)
    if args.num_users == client_num_per_round:
        client_indexes = [client_index for client_index in range(args.num_users)]
    else:
        num_clients = min(client_num_per_round, args.num_users)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(int(args.num_users)), int(num_clients), replace=False)
    # logger.info("client_indexes = %s" % str(client_indexes))
    return client_indexes


def test_client(args, dataset, test_idx, net):
    '''

    Test the performance of the client models on their datasets

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : The data on which we want the performance of the model to be evaluated

        args (dictionary) : The list of arguments defined by the user

        test_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local dataset of this client

    Returns:

        accuracy (float) : Percentage accuracy on test set of the model

        test_loss (float) : Cumulative loss on the data

    '''

    data_loader = DataLoader(DatasetSplit(dataset, test_idx), batch_size=args.local_bs)
    net.eval()
    # print (test_data)
    test_loss = 0
    correct = 0

    l = len(data_loader)

    with torch.no_grad():

        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs, _ = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs[0], target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs[0].data.max(1, keepdim=True)[1]

            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss

def cal_clip(w):
    norm = 0.0
    for name in w.keys():
        if (
                "running_mean" not in name
                and "running_var" not in name
                and "num_batches_tracked" not in name
        ):
            norm += pow(w[name].float().norm(2), 2)
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)
    # print(total_norm[0])
    return total_norm[0]


def clip_and_add_noise_our(w, args):
    l2_norm = cal_clip(w)
    print(l2_norm)
    # with torch.no_grad():
    #     for name in w.keys():
    #         if (
    #                 "running_mean" not in name
    #                 and "running_var" not in name
    #                 and "num_batches_tracked" not in name
    #         ):
    #             noise = torch.FloatTensor(w[name].shape).normal_(0, args.noise_multiplier * args.dp_clip / np.sqrt(
    #                 args.num_users * args.frac))
    #             noise = noise.cpu().numpy()
    #             noise = torch.from_numpy(noise).type(torch.FloatTensor).to(w[name].device)
    #             w[name] = w[name].float() * min(1, args.dp_clip / l2_norm)
    #             w[name] = w[name].add_(noise)
    return w


def get_sigma_or_epsilon(round, args):
    """DP setting"""
    delta = 1 / args.num_users ** 1.1  # DP budget
    c = args.dp_clip
    # “overhead”机制下，从epsilon计算噪声因子
    if args.dp_mode == "overhead":
        if args.dp_epsilon != -1:
            epsilon = args.dp_epsilon  # DP budget
            noise_multiplier = get_noise_multiplier_from_epsilon(
                # 隐私预算
                epsilon=epsilon,
                # 总步数，CL-DP中是通信轮数
                steps=args.epochs,
                # 每轮客户端采样率
                sample_rate=args.frac,
                # 给定的delta
                delta=delta,
                # 采用什么隐私机制
                mechanism=args.accountant,
            )
        else:
            noise_multiplier = args.noise_multiplier
            epsilon = get_epsilon(
                steps=args.epochs,
                noise_multiplier=args.noise_multiplier,
                sample_rate=args.frac,
                delta=1 / args.num_users ** 1.1,
                mechanism=args.accountant,
            )
            # epsilon = get_epsilon(
            #     steps=100,
            #     noise_multiplier=0.5,
            #     sample_rate=1,
            #     delta=1/10 ** 1.1,
            #     mechanism=args.accountant,
            # )
        # 计算加噪量，sigma
        sigma_averaged = (
                noise_multiplier * c / math.sqrt(args.num_users * args.frac)
        )
        # gaicheng log
        # with open(filename, 'a') as f:
        #     f.write('After Round{},Using noise multiplier = {:.4f} and sigma = {:.4f} to satisfy({:.4f}, {:.4f}) - DP.\n'.format(round,noise_multiplier,sigma_averaged,epsilon,delta))
    # "bound"机制下,epsilon和噪声因子都得给定
    elif args.dp_mode == "bounded":
        assert (
                args.dp_epsilon is not None and args.noise_multiplier is not None
        ), "To use bounded dp mode, noise_multiplier and target epsilon must both be specified."
        sigma_averaged = 0
        epsilon_cur = 0
        # with open(filename, 'a') as f:
        #     f.write('Bounding ({:.4f},{:.4f})-DP with noise multiplier={:.10f}.\n'.format(args.dp_epsilon,args.dp_delta,args.noise_multiplier))
    else:
        raise NotImplementedError
    return epsilon, noise_multiplier


def subtract(params_a, params_b):
    w = copy.deepcopy(params_a)
    if len(w.keys()) == len(params_b.keys()):
        for k in w.keys():
            w[k] = w[k] - params_b[k]
    else:
        min_len = min(len(w.keys()), len(params_b.keys()))
        for i, k in enumerate(w.keys()):
            if i < min_len:
                w[k] = w[k] - params_b[k]
            else:
                w[k] = w[k]
    return w


def add(params_a, params_b):
    w = copy.deepcopy(params_a)

    if len(w.keys()) == len(params_b.keys()):
        for k in w.keys():
            w[k] = w[k] + params_b[k]
    else:
        min_len = min(len(w.keys()), len(params_b.keys()))
        max_len = max(len(w.keys()), len(params_b.keys()))
        for i, k in enumerate(w.keys()):
            if i < min_len:
                w[k] = w[k] + params_b[k]
            else:
                w[k] = w[k]
    return w