import os
import time
import argparse
import asyncio
import numpy as np
import torch
import torch.optim as optim
from config import ClientConfig, CommonConfig
from comm_utils import *
from training_utils import train, test
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

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

parser.add_argument('--fune_type', type=str, choices=["FT", "FLoRA_QKV", "PLoRA_QKV"])
parser.add_argument('--local_step', type=int, default=-1)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

device = f"cuda:{rank % torch.cuda.device_count()}"


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
    common_config.decay_rate = client_config.common_config.decay_rate
    common_config.min_lr=client_config.common_config.min_lr
    common_config.epoch = client_config.common_config.epoch
    common_config.momentum = client_config.common_config.momentum
    common_config.weight_decay = client_config.common_config.weight_decay
    common_config.data_path = client_config.common_config.data_path
    common_config.para=client_config.para
    
    common_config.tag = 1
    ################################### init config #######################################
    # TODO: make it an args
    pretrained_model_path = "/data/jliu/models/bert-base-uncased"
    train_data_path = "/data/jliu/data/glue_data/SST-2/train.tsv"
    test_data_path = "/data/jliu/data/glue_data/SST-2/dev.tsv"

    ################################## init local model ###################################### 
    from mymodels import SST
    model = SST(pretrained_model_path)
    
    model.to(device)
    model.train()
    # TODO:according to different method, set the trainable parameters of the model

    logger.info(f"The model architecture --> ")
    trainable_paras = 0
    all_paras = 0
    for layer, para in model.named_parameters():
        all_paras += para.numel()
        if para.requires_grad:
            logger.info(f"{layer} --> para.shape = {para.shape}")
            trainable_paras += para.numel()
    logger.info(f"Trainable paras: {trainable_paras}, all paras: {all_paras} ---> {trainable_paras / all_paras}")
    
    ########################################## init training and test set ##################################
    from mydatasets import SST_reader
    train_dataset = SST_reader(train_data_path, 65)
    test_dataset = SST_reader(test_data_path, 65)

    optimizer=torch.optim.AdamW(model.parameters(),2e-5) # 2e-5 92.5；3e-5 91.5; 4e-5 91.1; 5e-5 90.5
    batch_size = 32
    loader_train=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    loader_test=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    
    ######################################### begin training #################################################
    logger.info("***** Running training *****")
    
    while True:
        # 开始本地训练
        logger.info(f"##################### Round {common_config.tag} start ... #########################")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                # train_and_eval(config,model,train_loader,tokenizer, rank, logger)
                # local_procedure(comm, common_config, config, model, train_loader, tokenizer, rank, logger)
                ada_lora_fl(comm, common_config, model, optimizer, loader_train, loader_test, logger)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        loop.close()
  
        common_config.tag += 1
        if common_config.tag==common_config.epoch+1:
            break

async def ada_lora_fl(comm, common_config, model, optimizer, loader_train, loader_test, logger):
    ######################### Trainer: local training ############################
    epoch = 0
    # Training
    for i in range(epoch):
        logger.info(f"################# training epoch {i} ###################")
        model.train()
        loss_sum=[]
        correct=0
        total=0
        for num,(label,mask,token) in enumerate(loader_train):
            label=label.to(device)
            mask=mask.to(device)
            token=token.to(device)
            pre,loss=model(label,mask,token)
            loss_sum.append(loss.item())
            pre=torch.argmax(pre,-1)
            correct+=(pre==label).sum().cpu().item()
            total+=label.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.info(f"training loss: {mean(loss_sum)}")
        logger.info(f"training accuracy: {correct/total}")
        model.eval()
        loss_sum=[]
        correct=0
        total=0
        with torch.no_grad():
            logger.info(f"evaluation...")
            for num, (label, mask, token) in enumerate(loader_test):
                label = label.to(device)
                mask = mask.to(device)
                token = token.to(device)
                pre, loss = model(label, mask, token)
                loss_sum.append(loss.item())
                pre = torch.argmax(pre, -1)
                correct += (pre == label).sum().cpu().item()
                total += label.shape[0]
            logger.info(f"test loss: {mean(loss_sum)}")
            logger.info(f"test accuracy: {str(correct / total)}")
    
    ######################### Maintainer: updating ###############################
    logger.info("Sending local parameters to the server")
    local_paras = dict()
    
    # 发送所有可以训练的参数
    trainable_paras = 0
    all_paras = 0
    for layer, paras in model.named_parameters():
        all_paras += paras.numel()
        if paras.requires_grad:
            local_paras[layer] = paras.clone().detach().to("cpu") # 都移动到cpu上方便聚合
            trainable_paras += paras.numel()
    
    await send_data(comm=comm, data=local_paras, dst_rank=MASTER_RANK, tag_epoch=common_config.tag)
    logger.info("Waiting and Receiving aggregated paras from the server...")
    received_paras = await get_data(comm=comm, src_rank=MASTER_RANK, tag_epoch=common_config.tag)
    logger.info("Updating local model with the received paras...")
    model_state_dict = model.state_dict()
    # update paras
    for layer, paras in received_paras.items():
        model_state_dict[layer] = paras.to(device)
    # TODO: according to the resource constraint, update trainable paras


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
    await send_data(comm=comm, data=local_paras, dst_rank=MASTER_RANK, tag_epoch=common_config.tag)
    logger.info("Waiting and Receiving aggregated paras from the server...")
    received_paras = await get_data(comm=comm, src_rank=MASTER_RANK, tag_epoch=common_config.tag)
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
