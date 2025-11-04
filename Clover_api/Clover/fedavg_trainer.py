import copy
import logging
import time
import pdb
import numpy as np
import torch
from torch import nn


# try:
from Clover_api.Clover.model_trainer import ModelTrainer
# except ImportError:
    # from FedML.Clover_core.trainer.model_trainer import ModelTrainer


class FedAvgTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger

    def set_masks(self, masks):
        self.masks=masks
       

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict= {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def train(self, train_data,  device,  args, round):
        # torch.manual_seed(0)
        model = self.model
        model.to(device)
        model.train()
        metrics = {
            'train_correct': 0,
            'train_loss': 0,
            'train_total': 0
        }
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr* (args.lr_decay**round), momentum=args.momentum,weight_decay=args.wd)
        
        for epoch in range(args.epochs):
            epoch_loss, epoch_acc = [], []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # model.zero_grad()
                
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())

                pred = model(x)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(labels).sum()
                epoch_acc.append(correct.item())

                metrics['train_correct'] += correct.item()
                metrics['train_loss'] += loss.item() * labels.size(0)
                metrics['train_total'] += labels.size(0)

        return metrics

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False




