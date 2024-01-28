import os
import time
import argparse
import asyncio
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import vector_to_parameters
from config import ClientConfig, CommonConfig
from comm_utils import *
from training_utils import train, test, training_step, eval_step, vallina_lora, add_adapter, customized_lora, set_trainble_para, customized_lora_avg
import mydatasets, mymodels
from mpi4py import MPI
import logging

from mymodels import BertForQA
from train_eval.train_eval import train_and_eval
from mydatasets import SQuAD_V2_Dataset
import timeit
from transformers.data.metrics.squad_metrics import compute_predictions_logits, compute_predictions_log_probs, squad_evaluate
from transformers.data.processors.squad import SquadResult,SquadFeatures,SquadExample
from train_eval.evaluate_official2 import eval_squad
import random
from config import BertForMRCConfig
from transformers import BertTokenizer, AutoTokenizer
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

from numpy import mean
from glue_utils import prepare_inputs


from transformers import BertTokenizerFast
from transformers.data.data_collator import DataCollatorWithPadding

import platform

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

parser.add_argument('--fune_type', type=str, choices=["FT", "FLoRA_QKV", "PLoRA_QKV"])
parser.add_argument('--local_step', type=int, default=-1)

parser.add_argument('--optimizer', type=str, default="SGD")

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

device = f"cuda:{rank % torch.cuda.device_count()}"


# if args.visible_cuda == '-1':
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(int(rank)% 4 + 0)
# else:
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_cuda
# device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


# init logger
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
RESULT_PATH = os.getcwd() + '/clients/' + now + '/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)


filename = RESULT_PATH + now + "_" +os.path.basename(__file__).split('.')[0] + '_'+ str(int(rank)) +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
# end logger

MASTER_RANK=0

async def get_init_config(comm, MASTER_RANK, config):
    config_received = await get_data(comm, MASTER_RANK, 1)
    for k, v in config_received.__dict__.items():
        setattr(config, k, v)


def main():
    
    
    global_round = 1
    
    # TODO: optimize only the trainable parameters
    # batch_size = 32
    # train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    # test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    hostname = platform.node() 
    if "407" in hostname:
        pretrained_model_path = "/data0/jliu/Models/LLM/bert-base-uncased"
    elif "406" in hostname:
        pretrained_model_path = "/data0/jliu/Models/bert-base-uncased"

    while True:
        ## 接收client的配置和最新的全局模型
        
        logger.info(f"##################### Round {global_round} start ... #########################")

        logger.info("client_rank:{}".format(rank))
        client_config = ClientConfig(
            common_config=CommonConfig()
        )
        
        logger.info("Receiving init config from the server...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        task = asyncio.ensure_future(
            get_init_config(comm,MASTER_RANK,client_config)
            )
        tasks.append(task)
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        logger.info("init config acknowledged.")
        
        common_config = CommonConfig()
        common_config.model_type = client_config.common_config.model_type
        common_config.dataset_type = client_config.common_config.dataset_type
        common_config.batch_size = client_config.common_config.batch_size
        common_config.data_pattern=client_config.common_config.data_pattern
        common_config.lr = client_config.common_config.lr
        logger.info(f"common_config.lr --> {common_config.lr}")
        common_config.decay_rate = client_config.common_config.decay_rate
        common_config.min_lr=client_config.common_config.min_lr
        common_config.epoch = client_config.common_config.epoch
        common_config.momentum = client_config.common_config.momentum
        common_config.weight_decay = client_config.common_config.weight_decay
        common_config.data_path = client_config.common_config.data_path
        common_config.para=client_config.para
        common_config.fedlora_rank = client_config.common_config.fedlora_rank
        common_config.fedlora_depth = client_config.common_config.fedlora_depth

        common_config.finetune_type = client_config.common_config.finetune_type

        common_config.heterlora_max_rank = client_config.common_config.heterlora_max_rank
        common_config.heterlora_min_rank = client_config.common_config.heterlora_min_rank

        common_config.client_rank = client_config.heterlora_client_rank
        common_config.our_total_rank = client_config.common_config.our_total_rank

        common_config.fedadpter_width = client_config.common_config.fedadpter_width
        common_config.fedadpter_depth = client_config.common_config.fedadpter_depth

        common_config.enable_sys_heter = client_config.common_config.enable_sys_heter
        common_config.test_target_matrix = client_config.common_config.test_target_matrix
        memory = client_config.memory
        
        local_steps = client_config.local_steps
        cur_steps = client_config.cur_steps

        logger.info(f"memory capacity --> {memory} GiB")
        
        logger.info(f"worker current runing client is --> {client_config.client_idx}")
        
        logger.info(f"client {client_config.client_idx} cur steps = {cur_steps} and local step is {local_steps}")
        

        from mymodels import CustomBERTModel
        if common_config.dataset_type == "ag_news":
            num_labels = 4
        elif common_config.dataset_type == "20news":
            num_labels = 20
        else:
            num_labels = 3 if common_config.dataset_type.startswith("mnli") else 1 if common_config.dataset_type=="stsb" else 2
        model = CustomBERTModel(pretrained_model_path, num_labels=num_labels, task=common_config.dataset_type)

        trainable = True
        logger.info(f"common_config.enable_sys_heter --> {common_config.enable_sys_heter}")
        if common_config.finetune_type == "fedft":
            if common_config.enable_sys_heter and memory < 12:
                # untrainable
                trainable = False
        elif common_config.finetune_type == "fedlora":
            model = vallina_lora(model, depth=common_config.fedlora_depth, rank=common_config.fedlora_rank, alpha=common_config.fedlora_rank * 2, test_target_matrix= common_config.test_target_matrix)
            if common_config.enable_sys_heter and memory < 6:
                # untrainable
                trainable = False
        elif common_config.finetune_type == "fedadapter":
            model = add_adapter(model, width=common_config.fedadpter_width, depth=common_config.fedadpter_depth)
            if common_config.enable_sys_heter and memory < 6:
                # untrainable
                trainable = False
        elif common_config.finetune_type == "our":
            model = customized_lora(model,common_config.our_total_rank, memory)
        elif common_config.finetune_type == "our_avg":
            logger.info(f"common_config.our_total_rank = {common_config.our_total_rank}")
            model = customized_lora_avg(model,common_config.our_total_rank, memory)
        elif common_config.finetune_type == "heterlora":
            logger.info(f"clint's heterlora_rank --> {common_config.client_rank}")
            model = vallina_lora(model, rank=common_config.client_rank, alpha=common_config.client_rank * 2)
            if common_config.enable_sys_heter and memory < 6:
                # untrainable
                trainable = False
        else:
            raise NotImplementedError
        
        
        logger.info(f"Is this client trainable? --> {trainable}")

        # heterlora truncate
        if common_config.finetune_type == "heterlora":
            logger.info("truncate tensors")
            for layer, paras in common_config.para.items():
                if "lora_A" in layer:
                    # logger.info(f"before truncate {layer}, shape = {paras.shape}")
                    common_config.para[layer] = paras[:common_config.client_rank, :]
                    # logger.info(f"after truncate {layer}, shape = {paras.shape}")
                if "lora_B" in layer:
                    # logger.info(f"before truncate {layer}, shape = {paras.shape}")
                    common_config.para[layer] = paras[:, : common_config.client_rank]
                    # logger.info(f"after truncate {layer}, shape = {paras.shape}")

        model.load_state_dict(common_config.para)

        model.to(device)
        # TODO:according to different method, set the trainable parameters of the model
        
        if args.optimizer == "SGD":
            optimizer=torch.optim.SGD(model.parameters(),common_config.lr) # 2e-5 92.5；3e-5 91.5; 4e-5 91.1; 5e-5 90.5
        elif args.optimizer == "Adam":
            optimizer=torch.optim.Adam(model.parameters(),common_config.lr)
        elif args.optimizer == "AdamW":
            optimizer=torch.optim.AdamW(model.parameters(),common_config.lr)
        else:
            raise NotImplementedError

        logger.info(f"common_config.finetune_type --> {common_config.finetune_type}")

        logger.info(f"Trainable parameters info --> ")
        trainable_paras = 0
        all_paras = 0
        for layer, para in model.named_parameters():
            all_paras += para.numel()
            if para.requires_grad:
                logger.info(f"{layer} --> para.shape = {para.shape}")
                trainable_paras += para.numel()

            if "out_layer" in layer:
                logger.info(f"{layer} --> {para}")
                trainable_paras += para.numel()
        logger.info(f"Trainable paras: {trainable_paras}, all paras: {all_paras} ---> {trainable_paras / all_paras}")
        
        ########################################## init training and test set ##################################
        train_dataset = client_config.source_train_dataset
        test_dataset = client_config.test_dataset

        logger.info(f"len(client_config.train_data_idxes) = {len(client_config.train_data_idxes)}")
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_path, use_fast=True)
        data_collator = DataCollatorWithPadding(tokenizer)
        logger.info(f"client_config.train_data_idxes --> {client_config.train_data_idxes}")
        train_loader = mydatasets.create_dataloaders(train_dataset, batch_size=common_config.batch_size, shuffle=False, selected_idxs=client_config.train_data_idxes, collate_fn=data_collator)
        test_loader = mydatasets.create_dataloaders(test_dataset, batch_size=common_config.batch_size, shuffle=False, collate_fn=data_collator)

        ##################################################### Local Training ######################################
        # 开始本地训练

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                # train_and_eval(config,model,train_loader,tokenizer, rank, logger)
                # local_procedure(comm, common_config, config, model, train_loader, tokenizer, rank, logger)
                ada_lora_fl(comm, common_config, model, optimizer, train_loader, len(client_config.train_data_idxes), test_loader, logger, trainable, memory, global_round, local_steps, cur_steps)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        loop.close()
  
        global_round += 1
        if global_round==common_config.epoch+1:
            break

async def ada_lora_fl(comm, common_config: CommonConfig, model, optimizer, train_loader, num_train, test_loader, logger, trainable, memory, global_round, local_steps, cur_steps):
    ######################### Trainer: local training ############################
    logger.info(f"the number of train data: {num_train}, batch size: {common_config.batch_size}")

    ######### evaluation before training #########
    # evaluation
    # logger.info("evaluation before training")
    # iterator = iter(test_loader)
    # trange = range(len(test_loader))
    # model.eval()
    # loss_all=[]
    # metric_name = model.metric.name
    # metric_1_name = None if model.metric_1 is None else model.metric_1.name
    # metric_all=[]
    # metric_1_all = []
    # for step in trange:
    #     inputs = prepare_inputs(next(iterator), device)
    #     step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
    #     loss_all.append(step_loss.item())
    #     metric_all.append(step_metric[model.metric.name])
    #     if model.metric_1 is not None: 
    #         metric_1_all.append(step_metric_1[model.metric_1.name])
            
    # logger.info(f"test loss --> {mean(loss_all)}")
    # logger.info(f"test {metric_name} --> {mean(metric_all)} ")
    # if model.metric_1 is not None:
    #     logger.info(f"test {metric_1_name} -->  {mean(metric_1_all)}")

    ##########################
    logger.info(f"trainbale: {trainable}")
    if trainable:
        logger.info(f"local steps --> {local_steps}")
        # Training
        iterator = iter(train_loader)
        # 先跳过之前记录的位置
        for step in range(cur_steps):
            try:
                next_data = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                next_data = next(iterator)
        # 再开始训练
        model.train()
        loss_all=[]
        metric_name = model.metric.name
        metric_1_name = None if model.metric_1 is None else model.metric_1.name
        metric_all=[]
        metric_1_all = []
        for step in range(local_steps):
            try:
                next_data = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                next_data = next(iterator)
            inputs = prepare_inputs(next_data, device)
            step_loss, step_metric, step_metric_1 = training_step(model, inputs, optimizer)
            loss_all.append(step_loss.item())
            metric_all.append(step_metric[model.metric.name])
            if model.metric_1 is not None: 
                metric_1_all.append(step_metric_1[model.metric_1.name])

        logger.info(f"train loss --> {mean(loss_all)}")
        logger.info(f"train {metric_name} --> {mean(metric_all)} ")
        if model.metric_1 is not None:
            logger.info(f"train {metric_1_name} -->  {mean(metric_1_all)}")
            
        # evaluation
        # test_iterator = iter(test_loader)
        # trange = range(len(test_loader))
        # model.eval()
        # loss_all=[]
        # metric_name = model.metric.name
        # metric_1_name = None if model.metric_1 is None else model.metric_1.name
        # metric_all=[]
        # metric_1_all = []
        # for step in trange:
        #     inputs = prepare_inputs(next(test_iterator), device)
        #     step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
        #     loss_all.append(step_loss.item())
        #     metric_all.append(step_metric[model.metric.name])
        #     if model.metric_1 is not None: 
        #         metric_1_all.append(step_metric_1[model.metric_1.name])
                
        # logger.info(f"test loss --> {mean(loss_all)}")
        # logger.info(f"test {metric_name} --> {mean(metric_all)} ")
        # if model.metric_1 is not None:
        #     logger.info(f"test {metric_1_name} -->  {mean(metric_1_all)}")
        
        ######################### Maintainer: updating ###############################
        logger.info("Sending local parameters to the server")
        local_paras = dict()
        
        # 发送所有可以训练的参数
        trainable_paras = 0
        all_paras = 0
        logger.info("Uploading local paras to server...")
        for layer, paras in model.named_parameters():
            all_paras += paras.numel()
            if paras.requires_grad:
                # logger.info(f"\t{layer} --> {paras}")
                local_paras[layer] = paras.clone().detach().to("cpu") # 都移动到cpu上方便聚合
                if common_config.finetune_type == "heterlora":
                    # padding 0
                    if "lora_A" in layer:
                        from torch.nn import functional as F
                        # logger.info(f"local_paras[layer] shape = {local_paras[layer].shape}")
                        padded_t = F.pad(local_paras[layer], (0,0,0,common_config.heterlora_max_rank-common_config.client_rank), value=0)  
                        local_paras[layer] = padded_t
                    if "lora_B" in layer:
                        padded_t = F.pad(local_paras[layer], (0, common_config.heterlora_max_rank-common_config.client_rank), value=0)
                        local_paras[layer] = padded_t
                trainable_paras += paras.numel()
        # logger.info("uploading dict --> ")
        # for layer, para in local_paras.items():
        #     logger.info(f"layer {layer}, shape = {para.shape}")
        
        await send_data(comm=comm, data=local_paras, dst_rank=MASTER_RANK, tag_epoch=global_round)
        # logger.info("Waiting and Receiving aggregated paras from the server...")
        # received_paras = await get_data(comm=comm, src_rank=MASTER_RANK, tag_epoch=global_round)
        # logger.info("Updating local model with the received paras...")
        # model_state_dict = model.state_dict()
        # # update paras
        #     ## heterlora truncate
        # if common_config.finetune_type == "heterlora":
        #     for layer, paras in received_paras.items():
        #         if "lora_A" in layer:
        #             received_paras[layer] = paras[:common_config.client_rank, :]
        #         if "lora_B" in layer:
        #             received_paras[layer] = paras[:, : common_config.client_rank]

        # logger.info(f"recevied paras from server...")
        # for layer, paras in received_paras.items():
        #     # logger.info(f"\t{layer} --> paras")
        #     layer_trainable = model_state_dict[layer].requires_grad
        #     model_state_dict[layer] = paras.to(device)
        #     model_state_dict[layer].requires_grad = layer_trainable
        # according to the resource constraint, update trainable paras
        # set_trainble_para(model, memory)
        # logger.info("local model after updating: ")
        # for layer, paras in model.named_parameters():
        #     logger.info(f"\t{layer} --> {paras}")
    
    else:
        logger.info("Sending empty parameters to the server")
        local_paras = dict()
        await send_data(comm=comm, data=local_paras, dst_rank=MASTER_RANK, tag_epoch=global_round)
        logger.info("Waiting and Receiving aggregated paras from the server...")
        received_paras = await get_data(comm=comm, src_rank=MASTER_RANK, tag_epoch=global_round)
        logger.info("Updating local model with the received paras...")
        model_state_dict = model.state_dict()
        # update paras
            ## heterlora truncate
        if common_config.finetune_type == "heterlora":
            for layer, paras in received_paras.items():
                if "lora_A" in layer:
                    received_paras[layer] = paras[:common_config.client_rank, :]
                if "lora_B" in layer:
                    received_paras[layer] = paras[:, : common_config.client_rank]

        for layer, paras in received_paras.items():
            layer_trainable = model_state_dict[layer].requires_grad
            model_state_dict[layer] = paras.to(device)
            model_state_dict[layer].requires_grad = layer_trainable
    


async def local_procedure(comm, common_config, config:BertForMRCConfig, model:PeftModel, train_loader, tokenizer, rank, logger):
    train_and_eval(config,model,train_loader,tokenizer, rank, logger, common_config)
    # 本地训练结束，上传lora参数
    logger.info("Sending local parameters to the server")
    local_paras = dict()
    # 不同的策略发送的参数不同
    if args.fune_type == "FT":
        # FT全模型发送
        for layer, paras in model.named_parameters():
            local_paras[layer] = paras.clone().detach().to("cpu") # 都移动到cpu上方便聚合
    else:
        for layer, paras in model.named_parameters():
            if "lora" in layer:
                local_paras[layer] = paras.clone().detach().to("cpu") # 都移动到cpu上方便聚合
    await send_data(comm=comm, data=local_paras, dst_rank=MASTER_RANK, tag_epoch=global_round)
    logger.info("Waiting and Receiving aggregated paras from the server...")
    received_paras = await get_data(comm=comm, src_rank=MASTER_RANK, tag_epoch=global_round)
    logger.info("Updating local model with the received paras...")
    model_state_dict = model.state_dict()
    for layer, paras in received_paras.items():
        model_state_dict[layer] = paras.to(config.device)


async def local_training(comm, common_config, train_loader, test_loader):
    local_model = mymodels.create_model_instance(common_config.dataset_type, common_config.model_type)
    torch.nn.utils.vector_to_parameters(common_config.para, local_model.parameters())
    local_model.to(device)
    epoch_lr = common_config.lr
    
    local_steps = 20
    if common_config.tag > 1 and common_config.tag % 1 == 0:
        epoch_lr = max((common_config.decay_rate * epoch_lr, common_config.min_lr))
        common_config.lr = epoch_lr
    logger.info("epoch-{} lr: {}".format(common_config.tag, epoch_lr))
    if common_config.momentum<0:
        optimizer = optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=common_config.weight_decay)
    else:
        optimizer = optim.SGD(local_model.parameters(),momentum=common_config.momentum, lr=epoch_lr, weight_decay=common_config.weight_decay)
    train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device, model_type=common_config.model_type)
    test_loss, acc = test(local_model, test_loader, device, model_type=common_config.model_type)
    logger.info("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(common_config.tag, train_loss, test_loss, acc))
    logger.info("send para")
    local_paras = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach()
    # send_data_await(comm, local_paras, MASTER_RANK, common_config.tag)
    await send_data(comm, local_paras, MASTER_RANK, common_config.tag)
    logger.info("after send")
    local_para = await get_data(comm, MASTER_RANK, common_config.tag)
    common_config.para=local_para
    common_config.tag = common_config.tag+1
    logger.info("get end")


if __name__ == '__main__':
    main()
