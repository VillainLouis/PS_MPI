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

def customized_lora(model, all_rank, memory):
    def findMaxLowerPowerOf2(n):
        power = math.floor(math.log2(n))
        return 1 << (power - 1)

    def alg(all_rank, max_len):
        ans = list()
        while all_rank > 2:
            ans.append(findMaxLowerPowerOf2(all_rank))
            all_rank -= ans[-1]
            if len(ans) == max_len:
                return ans
        if all_rank == 2:
            ans.append(all_rank) 
        return ans
    ranks = alg(all_rank, 6)
    # print(f"ranks --> {ranks}")
    layer_rank = dict()
    target_attn_matrix = dict()
    target_ffn_matrix = dict()

    last_layer_idx = 11
    for idx, r in enumerate(ranks):
        layer_rank[str(last_layer_idx - idx)] = r
        target_attn_matrix[str(last_layer_idx - idx)] = ["query", "key", "value", "output"]
        target_ffn_matrix[str(last_layer_idx - idx)] = ["intermediate", "output"]

    only_lora_B = False
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            rank = layer_rank[layer]
            alpha = 2 * rank
            # set attention.output
            if matrix == "output":
                module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer
            else:
                module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            

    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            rank = layer_rank[layer]
            module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
            if only_lora_B:
                lora_layer.lora_A.requires_grad = False
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
    lora.mark_only_lora_as_trainable(model)
    
    return set_trainble_para(model, memory)

def set_trainble_para(model, memory):
    # set layers according to memory
    if memory == 2:
        for layer, para in model.named_parameters():
            if "11" in layer:
                if "lora" in layer:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                para.requires_grad = False
    elif memory == 4:
        for layer, para in model.named_parameters():
            if "10" in layer or "11" in layer:
                if "lora" in layer:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                para.requires_grad = False
    elif memory == 6:
        for layer, para in model.named_parameters():
            if "8" in layer or "9" in layer or "10" in layer or "11" in layer:
                if "lora" in layer:
                    para.requires_grad = True
                else:
                    para.requires_grad = False
            else:
                para.requires_grad = False
    else:
        pass
    # 设置head可训练
    model._modules["linear"].weight.requires_grad = True
    model._modules["linear"].bias.requires_grad = True
    
    return model

def add_adapter(model, width = 32, depth = 12):
    def make_only_adapter_trainable(model):
        for layer, para in model.named_parameters():
            if "adapter" in layer:
                para.requires_grad = True
            else:
                para.requires_grad = False

    from torch import nn
    class Adapter(nn.Module):
        def __init__(self, input_dim, bottleneck_dim):
            super().__init__()
            self.down_project = nn.Linear(input_dim, bottleneck_dim, bias=False) 
            self.activation = nn.ReLU()  
            self.up_project = nn.Linear(bottleneck_dim, input_dim, bias=False)
            
        def forward(self, x):
            x = self.down_project(x) 
            x = self.activation(x)
            x = self.up_project(x)
            return x
        
    layers = [str(l) for l in range(11, 11 - depth, -1)]
    for layer in layers:
        origin_layer = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["output"]._modules["LayerNorm"]
        from torch.nn import Sequential
        import copy
        new_layer = Sequential()
        new_layer.add_module(layer, copy.deepcopy(origin_layer))
        adapter = Adapter(input_dim=768, bottleneck_dim=width)
        new_layer.add_module('adapter', adapter)

        model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["output"]._modules["LayerNorm"] = new_layer

    make_only_adapter_trainable(model)
    # 设置head可训练
    model._modules["linear"].weight.requires_grad = True
    model._modules["linear"].bias.requires_grad = True

    return model


def vallina_lora(model, depth = 12, rank = 8, alpha = 32):
    ####################################################
    ranks = [rank for _ in range(depth)]
    # print(f"ranks --> {ranks}")
    layer_rank = dict()
    target_attn_matrix = dict()
    target_ffn_matrix = dict()

    last_layer_idx = 11
    for idx, r in enumerate(ranks):
        layer_rank[str(last_layer_idx - idx)] = r
        target_attn_matrix[str(last_layer_idx - idx)] = ["query", "key", "value", "output"]
        target_ffn_matrix[str(last_layer_idx - idx)] = ["intermediate", "output"]

    only_lora_B = False
    for layer in target_attn_matrix.keys():
        for matrix in target_attn_matrix[layer]:
            rank = layer_rank[layer]
            alpha = 2 * rank
            # set attention.output
            if matrix == "output":
                module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules[matrix]._modules["dense"] = lora_layer
            else:
                module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
                if only_lora_B:
                    lora_layer.lora_A.requires_grad = False
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            

    for layer in target_ffn_matrix.keys():
        for matrix in target_ffn_matrix[layer]:
            rank = layer_rank[layer]
            module = model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
            lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=rank, lora_alpha=alpha)
            if only_lora_B:
                lora_layer.lora_A.requires_grad = False
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            model._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
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
