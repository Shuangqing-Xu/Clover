import argparse
import logging
import os
import random
import sys
import pdb
import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath("./Clover"))

from Clover_api.model.cv.lenet5 import LeNet5
from Clover_api.data_preprocessing.mnist.data_loader import load_partition_data_mnist
from Clover_api.data_preprocessing.fmnist.data_loader import load_partition_data_fmnist
from Clover_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from Clover_api.model.cv.resnet import  customized_resnet18, tiny_resnet18
from Clover_api.Clover.Clover_api import CloverAPI
from Clover_api.Clover.fedavg_trainer import FedAvgTrainer


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help="network architecture, supporting 'cnn_cifar10', 'cnn_cifar100', 'resnet18', 'vgg11'")

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='N',
                        help='momentum')

    parser.add_argument('--data_dir', type=str, default='data/data',
                        help='data directory, please feel free to change the directory to the right place')

    parser.add_argument('--partition_method', type=str, default='dir', metavar='N',
                        help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
                             "one called 'n_cls' short for how many classes allocated for each client"
                             "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution")

    parser.add_argument('--partition_alpha', type=float, default=0.6, metavar='PA',
                        help='available parameters for data partition method')

    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='local batch size for training')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')
    
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--lr_decay', type=float, default=0.998, metavar='LR_decay',
                        help='learning rate decay (default: 0.998)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=30, metavar='EP',
                        help='local training epochs for each client')
    
    parser.add_argument("--method", default='FedAvg', type=str, help="FedAvg")

    parser.add_argument('--client_num_in_total', type=int, default=500, metavar='NN',
                        help='the number of clients')

    parser.add_argument('--frac', type=float, default=0.1, metavar='NN',
                        help='the selection fraction of total clients in each round')

    parser.add_argument('--comm_round', type=int, default=1,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    parser.add_argument("--tag", type=str, default="test")

    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument('--sigma', type=float, default=0.95,
                        help='the standard deviation of client-level DP noise')
    parser.add_argument('--C', type=float, default=0.2,
                        help='the threshold of clipping in DP')
    parser.add_argument('--num_experiments', type=int, default=1,
                        help='the number of experiments')
    parser.add_argument('--alpha', type=float, default=0,
                        help='the sparsification ratio')
    return parser


def load_data(args, dataset_name):
    if dataset_name == "cifar10":
        args.data_dir += "cifar10"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_cifar10(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
    elif dataset_name == "mnist":
        
        args.data_dir += "mnist"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
    elif dataset_name == "fmnist":
        args.data_dir += "fmnist"
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_fmnist(args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, logger)
        

    
    dataset = [train_data_num, test_data_num,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset



def create_model(args, model_name,class_num,logger):
    # logger.info("create_model. model_name = %s" % (model_name))
    model = None
    if model_name == "lenet5":
        model = LeNet5(class_num)
    elif model_name =="resnet18":
        model = customized_resnet18(class_num=class_num)
    return model


def custom_model_trainer(args, model, logger):
    return FedAvgTrainer(model, args, logger)

def count_parameters(model):
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    return params

def get_buffer_names(model):
    return [name for name, _ in model.named_buffers()]

if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    # print("torch version{}".format(torch.__version__))

    data_partition = args.partition_method
    if data_partition != "IID":
        data_partition += str(args.partition_alpha)
    args.identity = args.method+"-"+args.dataset+"-"+ data_partition+"-"
    args. client_num_per_round = int(args.client_num_in_total* args.frac)
    args.identity += args.model
    args.identity += "-C" + str(args.C)
    args.identity += "-round" + str(args.comm_round) + "-num_client" + str(args.client_num_in_total)
    args.identity += "-l_ep"+ str(args.epochs)
    args.identity += "-l_bc"+ str(args.batch_size)
    args.identity += "-alpha"+ str(args.alpha)

    
    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    log_path = os.path.join(cur_dir, 'LOG/' + args.dataset + '/' + args.identity + '.log')
    logger = logger_config(log_path='LOG/' + args.dataset + '/' + args.identity + '.log', logging_name=args.identity)

    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) )
    logger.info(device)
    logger.info("running at device{}".format(device))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    for exper_index in range(args.num_experiments):
        # load data
        dataset = load_data(args, args.dataset)
        # print(dataset)
        
        # create model.
        if args.dataset == "emnist":
            model = create_model(args, model_name=args.model, class_num= 62, logger = logger)
        else:
            model = create_model(args, model_name=args.model, class_num= 10, logger = logger)

        model_trainer = custom_model_trainer(args, model, logger)
        # logger.info(model)

        num_of_params = count_parameters(model)
        if args.alpha:
            num_of_sparse_params = int(args.alpha * num_of_params)
        else:
            num_of_sparse_params = 0 # full vector
        print(
            f'Parameter size --- {num_of_params} (alpha={args.alpha})')

        buffer_names = get_buffer_names(model)

        CloverAPI = CloverAPI(dataset, device, args, model_trainer, logger, buffer_names, num_of_sparse_params)
        CloverAPI.train(exper_index)

