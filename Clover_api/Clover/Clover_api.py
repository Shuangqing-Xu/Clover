import copy
import logging
import pickle
import random
import pdb
import numpy as np
import torch
from tqdm import tqdm
from Clover_api.model.cv.resnet import  customized_resnet18
from Clover_api.Clover.client import Client
import os
from collections import OrderedDict
import time

class CloverAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger, buffer_names, num_of_sparse_params):
        self.num_of_sparse_params = num_of_sparse_params
        self.buffer_names = buffer_names
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    
    
    def train(self, exper_index):
        global_weight = self.model_trainer.get_model_params()
        for k in global_weight.keys(): # convert to cuda
                global_weight[k] = global_weight[k].to(self.device)

        self.logger.info("################Exper times: {}".format(exper_index))
        for round_idx in tqdm(range(self.args.comm_round)):
            local_weights_diff_list = []
            last_global_weight = copy.deepcopy(global_weight)

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)

            # self.logger.info("client_indexes = " + str(client_indexes))
            loss_locals, acc_locals, total_locals = [], [], []

            norm_list = []
            loop_count = len(client_indexes)
            for cur_clnt in client_indexes:
                start_time = time.time()
                # update dataset
                client = self.client_list[cur_clnt]
                # local model training
                w_per, metrics = client.train(copy.deepcopy(global_weight), round_idx)
                # calculate the local update of each participated client
                local_weights_diff = copy.deepcopy(subtract(w_per, global_weight))
                noisy_local_weights_diff = copy.deepcopy(subtract(w_per, global_weight))
                for k in local_weights_diff.keys():
                    local_weights_diff[k] = local_weights_diff[k].to(self.device)
                
                if self.num_of_sparse_params != 0:
                    ###### top-k sparsification ##############
                    top_k_local_weights_diff, top_k_indices = zero_except_top_k_weights_gpu(
                    local_weights_diff, self.buffer_names, self.num_of_sparse_params, self.device)
                    ###### clipping top-k gradient ##############
                    for name in local_weights_diff.keys():
                        top_k_local_weights_diff[name] *= min(1, self.args.C/torch.norm(top_k_local_weights_diff[name], 2))
                    local_weights_diff_list.append((client.get_sample_number(), top_k_local_weights_diff))
                else:
                    for name in local_weights_diff.keys():
                        local_weights_diff[name] *= min(1, self.args.C/torch.norm(local_weights_diff[name], 2))
                    local_weights_diff_list.append((client.get_sample_number(), local_weights_diff))
                    
                loss_locals.append(metrics['train_loss'])
                acc_locals.append(metrics['train_correct'])
                total_locals.append(metrics['train_total'])
                
            self._train_on_sample_clients(loss_locals, acc_locals, total_locals, round_idx, len(client_indexes))
            weights_diffs_global = self.secure_aggregate(local_weights_diff_list) # For utility evaluation under DP, try "dp_aggregate" for simulation, or replace using other MPC-based module. 
            global_weight = copy.deepcopy(add(last_global_weight, weights_diffs_global))

            
            self._test_on_all_clients(global_weight, round_idx)

            print('global_test_loss={}'.format(self.stat_info["global_test_loss"][-1]))
            print('global_test_acc={}'.format(self.stat_info["global_test_acc"][-1]))
            self.logger.info("################Communication round : {}".format(round_idx))
            if round_idx % 200 == 0 or round_idx == self.args.comm_round-1:
                self.logger.info("################The final results, Experiment times: {}".format(exper_index))
                if self.args.dataset ==  "cifar10":
                    model = customized_resnet18(10)
                    model.load_state_dict(copy.deepcopy(global_weight)) 


            self.logger.info('global_train_loss={}'.format(self.stat_info["global_train_loss"]))
            self.logger.info('global_train_acc={}'.format(self.stat_info["global_train_acc"]))
            self.logger.info('global_test_loss={}'.format(self.stat_info["global_test_loss"]))
            self.logger.info('global_test_acc={}'.format(self.stat_info["global_test_acc"]))
                
        


    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def dp_aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        global_weight ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    global_weight[k] = local_model_params[k] * w
                    noise = torch.FloatTensor(global_weight[k].shape).normal_(0, self.args.sigma * self.args.C) /len(w_locals)
                    noise = noise.cpu().numpy()
                    noise = torch.from_numpy(noise).type(torch.FloatTensor).to(global_weight[k].device)
                    global_weight[k] += noise
                else:
                    global_weight[k] += local_model_params[k] * w
        return global_weight
    
    def secure_aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        global_weight ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    global_weight[k] = local_model_params[k] * w
                else:
                    global_weight[k] += local_model_params[k] * w
        return global_weight

    def _train_on_sample_clients(self, loss_locals, acc_locals, total_locals, round_idx, client_sample_number):
        self.logger.info("################global_train_on_all_clients : {}".format(round_idx))

        g_train_acc = sum([np.array(acc_locals[i]) / np.array(total_locals[i]) for i in
                        range(client_sample_number)]) / client_sample_number
        g_train_loss = sum([np.array(loss_locals[i]) / np.array(total_locals[i]) for i in
                         range(client_sample_number)]) / client_sample_number
        
        print(("################Communication round : {}".format(round_idx)))
        print('The averaged global_train_acc:{}, global_train_loss:{}'.format(g_train_acc, g_train_loss))
        stats = {'The averaged global_train_acc': g_train_acc, 'global_train_loss': g_train_loss}
        self.stat_info["global_train_acc"].append(g_train_acc)
        self.stat_info["global_train_loss"].append(g_train_loss)
        self.logger.info(stats)


    def _test_on_all_clients(self, global_weight, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))
        g_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        for client_idx in range(self.args.client_num_in_total):
            client = self.client_list[client_idx]
            g_test_local_metrics = client.local_test(global_weight, True)
            g_test_metrics['num_samples'].append(copy.deepcopy(g_test_local_metrics['test_total']))
            g_test_metrics['num_correct'].append(copy.deepcopy(g_test_local_metrics['test_correct']))
            g_test_metrics['losses'].append(copy.deepcopy(g_test_local_metrics['test_loss']))


            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on test dataset
        g_test_acc = sum([np.array(g_test_metrics['num_correct'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                        range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        g_test_loss = sum([np.array(g_test_metrics['losses'][i]) / np.array(g_test_metrics['num_samples'][i]) for i in
                         range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'global_test_acc': g_test_acc, 'global_test_loss': g_test_loss}
        self.stat_info["global_test_acc"].append(g_test_acc)
        self.stat_info["global_test_loss"].append(g_test_loss)
        self.logger.info(stats)



    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["global_train_acc"] = []
        self.stat_info["global_train_loss"] = []
        self.stat_info["global_test_acc"] = []
        self.stat_info["global_test_loss"] = []
        self.stat_info["person_test_acc"] = []
        self.stat_info["final_masks"] = []



def subtract(params_a, params_b):
        w = copy.deepcopy(params_a)
        for k in w.keys():
                w[k] -= params_b[k]
        return w

def add(params_a, params_b):
        w = copy.deepcopy(params_a)
        for k in w.keys():
                w[k] += params_b[k]
        return w

def get_index_ranges(learnable_parameters):
    """
    Args:
        learnable_parameters (OrderedDict): parameters without buffers (such as bn.running_mean)
    Returns:
        indices: [(int, int)]
            [(start, end)]
    """
    index_ranges = []
    s = 0
    for _, p in learnable_parameters.items():
        size = torch.flatten(p).shape[0]
        index_ranges.append((s, s + size))
        s += size
    return index_ranges

def get_learnable_parameters(state_dict, buffer_names):
        learnable_parameters = OrderedDict()
        for key, value in state_dict.items():
            if key not in buffer_names:
                learnable_parameters[key] = value
        return learnable_parameters

def flatten_params(learnable_parameters):
    """
    Args:
        learnable_parameters (OrderedDict): parameters without buffers (such as bn.running_mean)
    Returns:
        flat (torch.Tensor):
            whose dim is one, like [0.1, ..., 0.2]

    """
    ir = [torch.flatten(p) for _, p in learnable_parameters.items()]
    flat = torch.cat(ir).view(-1, 1).flatten()
    return flat

def recover_flattened(flat_params, base_state_dict, learnable_parameters):
    """
    Args:
        flat_params (torch.Tensor):
            whose dim is one, like [0.1, ..., 0.2]
        base_state_dict (OrderedDict)
            ex. model.state_dict():
            buffers are inherent
        learnable_parameters (OrderedDict):
            parameters without buffers (such as bn.running_mean)
    Returns:
        new_state: OrderedDict
            ex. model.state_dict()
    """
    index_ranges = get_index_ranges(learnable_parameters)
    ir = [flat_params[s:e] for (s, e) in index_ranges]
    new_state = copy.deepcopy(base_state_dict)
    for flat, (key, value) in zip(ir, learnable_parameters.items()):
        if len(value.shape) == 0:
            new_state[key] = flat[0]
        else:
            new_state[key] = flat.view(*value.shape)
    return new_state

def zero_except_top_k_weights_gpu(state_dict, buffer_names, k, device):
    learnable_parameters = get_learnable_parameters(state_dict, buffer_names)
    tensor_flat_params = flatten_params(learnable_parameters).to(device) 
    top_k_values, top_k_indices = torch.topk(torch.abs(tensor_flat_params), k)
    top_k_mask = torch.zeros_like(tensor_flat_params)
    top_k_mask[top_k_indices] = 1

    top_k_flat_params = tensor_flat_params * top_k_mask

    new_state = recover_flattened(top_k_flat_params, state_dict, learnable_parameters)
    return new_state, top_k_indices.tolist()

def noisy_zero_except_top_k_weights_gpu(state_dict, noisy_local_weights_diff, buffer_names, k, device):
    learnable_parameters = get_learnable_parameters(state_dict, buffer_names)
    tensor_flat_params = flatten_params(learnable_parameters).to(device)  
    
    noisy_learnable_parameters = get_learnable_parameters(noisy_local_weights_diff, buffer_names)
    noisy_tensor_flat_params = flatten_params(noisy_learnable_parameters).to(device)  
    top_k_values, top_k_indices = torch.topk(torch.abs(tensor_flat_params), k)
    top_k_mask = torch.zeros_like(tensor_flat_params)
    top_k_mask[top_k_indices] = 1

    top_k_flat_params = noisy_tensor_flat_params * top_k_mask

    new_state = recover_flattened(top_k_flat_params, state_dict, learnable_parameters)
    return new_state, top_k_indices.tolist()