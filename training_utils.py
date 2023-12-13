import sys
import time
import math
import re
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Union
import loralib as lora

def vallina_lora(model, device, rank = 8, alpha = 32):
    ####################################################
    target_attn_matrix = { # attn
            "0": ["query", "key", "value", "output"],
            "1": ["query", "key", "value", "output"],
            "2": ["query", "key", "value", "output"],
            "3": ["query", "key", "value", "output"],
            "4": ["query", "key", "value", "output"],
            "5": ["query", "key", "value", "output"],
            "6": ["query", "key", "value", "output"],
            "7": ["query", "key", "value", "output"],
            "8": ["query", "key", "value", "output"],
            "9": ["query", "key", "value", "output"],
            "10": ["query", "key", "value", "output"],
            "11": ["query", "key", "value", "output"]
        }
    target_ffn_matrix = { # ffn
        "0": ["intermediate", "output"],
        "1": ["intermediate", "output"],
        "2": ["intermediate", "output"],
        "3": ["intermediate", "output"],
        "4": ["intermediate", "output"],
        "5": ["intermediate", "output"],
        "6": ["intermediate", "output"],
        "7": ["intermediate", "output"],
        "8": ["intermediate", "output"],
        "9": ["intermediate", "output"],
        "10": ["intermediate", "output"],
        "11": ["intermediate", "output"]
    }
    only_lora_B = False
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            # set attention.output
            if matrix == "output":
                module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer.to(device)
            else:
                module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer.to(device)
            

    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
            if only_lora_B:
                lora_layer.lora_A.requires_grad = False
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer.to(device)

    lora.mark_only_lora_as_trainable(model)

    # 设置head可训练
    model._modules["linear"].weight.requires_grad = True
    model._modules["linear"].bias.requires_grad = True
    
    return model



def training_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                  optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        :obj:`torch.Tensor`: The tensor with training loss on this batch.
    """
    loss, metric, metric_1 = compute_loss(model, inputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()
    # lr_scheduler.step()

    return loss.detach(), metric, metric_1


def eval_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        :obj:`torch.Tensor`: The tensor with training loss on this batch.
    """
    model.eval()
    model.zero_grad()
    loss, metric, metric_1 = compute_loss(model, inputs)

    return loss.detach(), metric, metric_1

def compute_loss(model, inputs):
    """
    
    """
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)

    logits = outputs["logits"]
    loss = model.loss(logits, labels)
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)
    
    return (loss, metric, metric_1)


def train(model, data_loader, optimizer, local_iters=None, device=torch.device("cpu"), model_type=None):
    t_start = time.time()
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)
    #print("local_iters: ", local_iters)

    train_loss = 0.0
    samples_num = 0
    for iter_idx in range(local_iters):
        data, target = next(data_loader)

        if model_type == 'LR':
            data = data.squeeze(1).view(-1, 28 * 28)
            
        data, target = data.to(device), target.to(device)
        
        output = model(data)

        optimizer.zero_grad()
        
        loss_func = nn.CrossEntropyLoss() 
        loss =loss_func(output, target)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss /= samples_num
    
    return train_loss

def test(model, data_loader, device=torch.device("cpu"), model_type=None):
    model.eval()
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            if model_type == 'LR':
                data = data.squeeze(1).view(-1, 28 * 28)
            output = model(data)

            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum') 
            test_loss += loss_func(output, target).item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    # TODO: Record

    return test_loss, test_accuracy
