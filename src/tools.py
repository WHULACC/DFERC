#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import torch 
import numpy as np
import random
import os

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_paramsgroup(model, config, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = float(config['bert_lr'])

    bert_params = list(map(id, model.bert.parameters()))
    params = []
    warmup_params = []
    base_lr = float(config.lr)
    for name, param in model.named_parameters():
        lr = base_lr
        weight_decay = 0.01
        if id(param) in bert_params:
            lr = pre_train_lr
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params': param,
            'lr': pre_train_lr / 4 if id(param) in bert_params else lr,
            'weight_decay': weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: float(x['lr']))
    return params

def load_param(config, model, trainLoader):
        
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if 'bert' in n and not any(nd in n for nd in no_decay)], 'weight_decay': float(config.weight_decay), 'lr': float(config.bert_lr)},
            {'params': [p for n, p in param_optimizer if 'bert' in n and any(nd in n for nd in no_decay)], 'weight_decay': 0, 'lr': float(config.bert_lr)},
            {'params': [p for n, p in param_optimizer if 'bert' not in n and not any(nd in n for nd in no_decay)], 'weight_decay': float(config.weight_decay), 'lr': float(config.lr)},
            {'params': [p for n, p in param_optimizer if 'bert' not in n and any(nd in n for nd in no_decay)], 'weight_decay': 0, 'lr': float(config.lr)}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, betas=(config.beta1, config.beta2), eps=float(config.adam_epsilon))
    config.warmup_steps = 100
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                        num_training_steps=config.epoch_size * trainLoader.__len__())

    config.optimizer = optimizer
    config.scheduler = scheduler 

    return config

