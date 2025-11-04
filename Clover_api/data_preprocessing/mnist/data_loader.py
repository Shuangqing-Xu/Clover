import logging
import math
import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from .datasets import MNIST_truncated  
def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = []
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = []
        for i in range(10):
            if i in unq:
                tmp.append(unq_cnt[np.argwhere(unq == i)][0, 0])
            else:
                tmp.append(0)
        net_cls_counts.append(tmp)
    return net_cls_counts

def _data_transforms_mnist():  
    MNIST_MEAN = [0.1307]
    MNIST_STD = [0.3081]

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    return train_transform, valid_transform

def load_mnist_data(datadir):  
    train_transform, test_transform = _data_transforms_mnist()
    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target
    return (X_train, y_train, X_test, y_test)

def partition_data(datadir, partition, n_nets, alpha, logger): 
    logger.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_mnist_data(datadir)  
    n_train = X_train.shape[0]

    if partition == "IID":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    
    
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

def get_dataloader_mnist(datadir, train_bs, test_bs, dataidxs=None, test_idxs=None, 
                        cache_train_data_set=None, cache_test_data_set=None, logger=None):
    transform_train, transform_test = _data_transforms_mnist()
    dataidxs = np.array(dataidxs)
    
    train_ds = MNIST_truncated(datadir, dataidxs=dataidxs, train=True, 
                              transform=transform_train, download=True, 
                              cache_data_set=cache_train_data_set)
    test_ds = MNIST_truncated(datadir, dataidxs=test_idxs, train=False, 
                             transform=transform_test, download=True,
                             cache_data_set=cache_test_data_set)
    
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=True)
    return train_dl, test_dl

def load_partition_data_mnist(data_dir, partition_method, partition_alpha, 
                             client_number, batch_size, logger):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        data_dir, partition_method, client_number, partition_alpha, logger)
    
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    transform_train, transform_test = _data_transforms_mnist()
    cache_train_data_set = MNIST(data_dir, train=True, transform=transform_train, download=True)
    cache_test_data_set = MNIST(data_dir, train=False, transform=transform_test, download=True)
    idx_test = [[] for i in range(10)]
    # checking
    for label in range(10):
        idx_test[label] = np.where(y_test == label)[0]
    test_dataidxs = [[] for i in range(client_number)]
    tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    for client_idx in range(client_number):
        for label in range(10):
            # each has 100 pieces of testing data
            label_num = math.ceil(traindata_cls_counts[client_idx][label] / sum(traindata_cls_counts[client_idx]) * tmp_tst_num)
            rand_perm = np.random.permutation(len(idx_test[label]))
            if len(test_dataidxs[client_idx]) == 0:
                test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
            else:
                test_dataidxs[client_idx] = np.concatenate(
                    (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]]))
        dataidxs = net_dataidx_map[client_idx]
        train_data_local, test_data_local = get_dataloader_mnist( data_dir, batch_size, batch_size,
                                                 dataidxs,test_dataidxs[client_idx] ,cache_train_data_set=cache_train_data_set,cache_test_data_set=cache_test_data_set,logger=logger)
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
        # logger.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    
    return None, None, None, None, data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts