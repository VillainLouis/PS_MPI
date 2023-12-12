from fileinput import filename
import os
import argparse
import asyncio
import time
import numpy as np
from numpy import mean
import torch
from config import *
import torch.nn.functional as F
import mydatasets
import mymodels
from training_utils import test, eval_step

from mpi4py import MPI

import logging
import random

from noniid_label import label_skew_process
from glue_utils import prepare_inputs

#init parameters
parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--dataset_type', type=str, default='sst2')
parser.add_argument('--model_type', type=str, default='Bert')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_pattern', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--min_lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--momentum', type=float, default=-1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data_path', type=str, default='/data/jliu/data')
parser.add_argument('--use_cuda', action="store_false", default=True)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--finetune_type', type=str, choices=["fedft", "fedlora", "fedadapter", "our"])

args = parser.parse_args()
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

np.random.seed(args.seed)
random.seed(args.seed)

alpha = args.alpha
device = f"cuda:{rank % torch.cuda.device_count()}"

RESULT_PATH = os.getcwd() + '/server/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH, exist_ok=True)

# init logger
logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
logger.setLevel(logging.INFO)

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
filename = RESULT_PATH + now + "_" +os.path.basename(__file__).split('.')[0] +'.log'
fileHandler = logging.FileHandler(filename=filename)
formatter = logging.Formatter("%(message)s")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)


def main():
    logger.info("csize:{}".format(int(csize)))
    logger.info("server start (rank):{}".format(int(rank)))
    # init config
    common_config = CommonConfig()
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.data_pattern=args.data_pattern
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.min_lr=args.min_lr
    common_config.epoch = args.epoch
    common_config.momentum = args.momentum
    common_config.data_path = args.data_path
    common_config.weight_decay = args.weight_decay

    common_config.finetune_type = args.finetune_type

    worker_num = int(csize)-1

    ###################################### init config #############################################
    pretrained_model_path = "/data/jliu/models/bert-base-uncased"

    ###################################### init model ###############################################
    # from mymodels import SST
    # global_model = SST(pretrained_model_path)
    from mymodels import CustomBERTModel
    logger.info(f"\nLoading pre-trained BERT model \"{pretrained_model_path}\"")
    num_labels = 3 if common_config.dataset_type.startswith("mnli") else 1 if common_config.dataset_type=="stsb" else 2
    global_model = CustomBERTModel(pretrained_model_path, num_labels=num_labels, task=common_config.dataset_type)
    
    
    # global_model = mymodels.create_model_instance(common_config.dataset_type, common_config.model_type)
    common_config.para_nums = sum(p.numel() for p in global_model.parameters())
    model_size = common_config.para_nums * 4 / 1024 / 1024
    logger.info("para num: {}".format(common_config.para_nums))
    logger.info("Model Size: {} MB".format(model_size))

    # create workers
    worker_list: list[Worker] = list()
    for worker_idx in range(worker_num):
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config),rank=worker_idx+1)
        )
    #到了这里，worker已经启动了

    ###################################### init server side dataset (train, test) and set data partition ####################################
    from mydatasets import RandomPartitioner
    # train_dataset, test_dataset = mydatasets.load_datasets(common_config.dataset_type)
    if common_config.dataset_type in [ "cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2",  "stsb", "wnli"]:
        from mydatasets import get_glue_dataset
        train_dataset = get_glue_dataset(common_config.dataset_type, pretrained_model_path, "train", batch_size=common_config.batch_size)
        test_dataset = get_glue_dataset(common_config.dataset_type, pretrained_model_path, "validation", batch_size=common_config.batch_size)
        from transformers import BertTokenizerFast
        from transformers.data.data_collator import DataCollatorWithPadding
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_path, use_fast=True)
        data_collator = DataCollatorWithPadding(tokenizer)
    test_loader = mydatasets.create_dataloaders(test_dataset, batch_size=common_config.batch_size, shuffle=False, collate_fn=data_collator)

    # dataloader = DataLoader(
    #                 encoded_dataset[split],
    #                 shuffle=shuffle,
    #                 batch_size=batch_size,
    #                 collate_fn=data_collator,
    #                 drop_last=dataloader_drop_last,
    #                 num_workers=dataloader_num_workers,
    #                 pin_memory=dataloader_pin_memory,
    # )

    # use alpha to control the overall data partition
    if args.data_pattern != 0:
        logger.info("non-IID partition prepare...")
        if args.dataset_type == "sst2":    
            label_vocab = {'negative': 0, 'positive': 1}
            label_map = {0 : 'negative', 1 : 'positive'}
            label_assignment_train = np.array([
                label_map[int(train_dataset[idx]['label'])]
                for idx in range(len(train_dataset))
            ])
            train_data_partition = label_skew_process(label_vocab, label_assignment_train, worker_num, alpha, len(train_dataset), logger)
        else:
            raise NotImplementedError
    else:
        logger.info("IID partition...")
        train_data_partition = RandomPartitioner(data_len=len(train_dataset), partition_sizes=[1/worker_num for _ in range(worker_num)])
    
    # train_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern)

    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = global_model.state_dict()
        if args.data_pattern != 0:
            worker.config.train_data_idxes = train_data_partition[worker_idx]
        else:
            worker.config.train_data_idxes = train_data_partition.use(worker_idx)
        worker.config.source_train_dataset = train_dataset
        worker.config.test_dataset = test_dataset
        # logger.info(f"$$$$$$$$$$$ worker {worker_idx} --> {worker.config.train_data_idxes}")
    
    ###################### Sending init config to clients ###############################
    logger.info(f"Sending init config to all clients and start the training procedure")
    communication_parallel(worker_list, 1, comm, action="init")

    for epoch_idx in range(1, 1+common_config.epoch):
        logger.info(f"################## Round {epoch_idx} begin #####################")
        logger.info("Waiting and receiving updated paras from clients")
        communication_parallel(worker_list, epoch_idx, comm, action="get_para")
        logger.info("Clients' information received.")
        logger.info("Performing aggregation...")
        global_para = parameter_wise_aggregation(worker_list)
        logger.info("Aggregation finished and sending the newly aggregated paras back to clients")
        communication_parallel(worker_list, epoch_idx, comm, action="send_model",data=global_para)
        logger.info(f"TEST on server...")
        global_model_sd = global_model.state_dict()
        global_model_sd.update(global_para)
        global_model.load_state_dict(global_model_sd)
        global_model.to("cuda:0")
        global_model.eval()
        
        if common_config.dataset_type in [ "cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2",  "stsb", "wnli"]:
             # evaluation
            iterator = iter(test_loader)
            trange = range(len(test_loader))
            loss_all=[]
            metric_name = global_model.metric.name
            metric_1_name = None if global_model.metric_1 is None else global_model.metric_1.name
            metric_all=[]
            metric_1_all = []
            for step in trange:
                inputs = prepare_inputs(next(iterator), device)
                step_loss, step_metric, step_metric_1 = eval_step(global_model, inputs)
                loss_all.append(step_loss.item())
                metric_all.append(step_metric[global_model.metric.name])
                if global_model.metric_1 is not None: 
                    metric_1_all.append(step_metric_1[global_model.metric_1.name])
                
            logger.info(f"test loss --> {mean(loss_all)}")
            logger.info(f"test {metric_name} --> {mean(metric_all)} ")
            if global_model.metric_1 is not None:
                logger.info(f"test {metric_1_name} -->  {mean(metric_1_all)}")

        logger.info(f"Round {epoch_idx} finished")

    # close socket
    
def parameter_wise_aggregation(worker_list):
    overall_para = dict()
    with torch.no_grad():
        for worker_idx in range(len(worker_list)):
            for layer, paras in worker_list[worker_idx].config.neighbor_paras.items():
                if layer not in overall_para:
                    overall_para[layer] = dict()
                    overall_para[layer]['cnt'] = 1
                    overall_para[layer]['value'] = paras
                else:
                    overall_para[layer]['cnt'] += 1
                    overall_para[layer]['value'] += paras
    aggregate_para_dict = dict()
    for layer in overall_para.keys():
        aggregate_para_dict[layer] = overall_para[layer]['value'] / overall_para[layer]['cnt']
    return aggregate_para_dict

def aggregate_para_dict(worker_list):
    with torch.no_grad():
        aggregated_paras = worker_list[0].config.neighbor_paras
        for layer in aggregated_paras.keys():
            for worker in range(1, len(worker_list)):
                aggregated_paras[layer] += worker_list[worker].config.neighbor_paras[layer]
            aggregated_paras[layer] /= len(worker_list)
    return aggregated_paras
        
    

def aggregate_model_para(global_model, worker_list):
    global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    with torch.no_grad():
        para_delta = torch.zeros_like(global_para)
        for worker in worker_list:
            model_delta = (worker.config.neighbor_paras - global_para)
            #gradient
            # model_delta = worker.config.neighbor_paras
            para_delta += worker.config.average_weight * model_delta
        global_para += para_delta
    torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
    return global_para

def communication_parallel(worker_list, epoch_idx, comm, action, data=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for worker in worker_list:
        if action == "init":
            task = asyncio.ensure_future(worker.send_init_config(comm, epoch_idx))
        elif action == "get_para":
            task = asyncio.ensure_future(worker.get_model(comm, epoch_idx))
        elif action == "send_model":
            task = asyncio.ensure_future(worker.send_data(data, comm, epoch_idx))
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

    return partition_sizes

def partition_data(dataset_type, data_pattern, worker_num=10):
    train_dataset, _ = mydatasets.load_datasets(dataset_type=dataset_type,data_path=args.data_path)

    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num=10
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    elif dataset_type == "EMNIST":
        train_class_num=62
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    if dataset_type == "CIFAR100" or dataset_type == "image100":
        train_class_num=100
        if data_pattern == 0:
            partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 2:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 3:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
        elif data_pattern == 4:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio,train_class_num,worker_num)
    train_data_partition = mydatasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    return train_data_partition

if __name__ == "__main__":
    main()
