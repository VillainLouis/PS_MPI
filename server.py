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
from training_utils import test, eval_step, vallina_lora, add_adapter, customized_lora, customized_lora_avg

from mpi4py import MPI

import logging
import random

from noniid_label import label_skew_process
from glue_utils import prepare_inputs

import platform

import math

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

parser.add_argument('--fedlora_rank', type=int, default=4)
parser.add_argument('--fedlora_depth', type=int, default=12)

parser.add_argument('--finetune_type', type=str, choices=["fedft", "fedlora", "fedadapter", "our", "heterlora", "our_avg"])

parser.add_argument("--max_rank", type=int, default=64)
parser.add_argument("--min_rank", type=int, default=2)


parser.add_argument("--our_total_rank", type=int, default=192)

parser.add_argument("--fedadpter_width", type=int, default=32)
parser.add_argument("--fedadpter_depth", type=int, default=12)

parser.add_argument("--partitial_data", type=float, default=1.0)

parser.add_argument("--enable_sys_heter", type=bool, default=False)

parser.add_argument('--test_target_matrix', type=str, default=None)

parser.add_argument('--client_num', type=int, default=200)

args = parser.parse_args()
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

np.random.seed(args.seed)
random.seed(args.seed)

torch.manual_seed(args.seed)


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
    common_config.fedlora_rank = args.fedlora_rank
    common_config.fedlora_depth = args.fedlora_depth

    common_config.heterlora_max_rank = args.max_rank
    common_config.heterlora_min_rank = args.min_rank
    common_config.our_total_rank = args.our_total_rank
    common_config.fedadpter_width = args.fedadpter_width
    common_config.fedadpter_depth = args.fedadpter_depth
    common_config.enable_sys_heter = args.enable_sys_heter
    logger.info(f"system heter is enable ? --> {common_config.enable_sys_heter}")
    logger.info(f"data hetero is enable ? --> {common_config.data_pattern == 1}")
    common_config.test_target_matrix = args.test_target_matrix
    logger.info(f"test target marix = {common_config.test_target_matrix}")
    worker_num = int(csize)-1

    logger.info(f"learning rate: {common_config.lr}")
    ###################################### init config #############################################
    hostname = platform.node() 
    if "407" in hostname:
        pretrained_model_path = "/data0/jliu/Models/LLM/bert-base-uncased"
    elif "406" in hostname:
        pretrained_model_path = "/data0/jliu/Models/bert-base-uncased"

    ###################################### init model ###############################################
    # from mymodels import SST
    # global_model = SST(pretrained_model_path)
    from mymodels import CustomBERTModel
    logger.info(f"\nLoading pre-trained BERT model \"{pretrained_model_path}\"")
    if common_config.dataset_type == "ag_news":
        num_labels = 4
    elif common_config.dataset_type == "20news":
        num_labels = 20
    else:
        num_labels = 3 if common_config.dataset_type.startswith("mnli") else 1 if common_config.dataset_type=="stsb" else 2
    global_model = CustomBERTModel(pretrained_model_path, num_labels=num_labels, task=common_config.dataset_type)

    
    logger.info(f"strategy --> {common_config.finetune_type}")

    if common_config.finetune_type == "fedft":
        pass
    elif common_config.finetune_type == "fedlora":
        logger.info(f"fedlora_rank --> {args.fedlora_rank} fedlora_depth --> {common_config.fedlora_depth}")
        global_model = vallina_lora(global_model, depth=common_config.fedlora_depth, rank=args.fedlora_rank, alpha=args.fedlora_rank * 2, test_target_matrix=common_config.test_target_matrix)
    elif common_config.finetune_type == "fedadapter":
        logger.info(f"common_config.fedadpter_width = {common_config.fedadpter_width}, common_config.fedadpter_depth = {common_config.fedadpter_depth}")
        global_model = add_adapter(global_model, width=common_config.fedadpter_width, depth=common_config.fedadpter_depth)
    elif common_config.finetune_type == "our":
        logger.info(f"common_config.our_total_rank = {common_config.our_total_rank}")
        global_model = customized_lora(global_model,common_config.our_total_rank, memory=100)
    elif common_config.finetune_type == "our_avg":
        logger.info(f"common_config.our_total_rank = {common_config.our_total_rank}")
        global_model = customized_lora_avg(global_model,common_config.our_total_rank, memory=100)
    elif common_config.finetune_type == "heterlora":
        logger.info(f"heterlora_rank --> max: {common_config.heterlora_max_rank} min: {common_config.heterlora_min_rank}")
        global_model = vallina_lora(global_model, rank=common_config.heterlora_max_rank, alpha=common_config.heterlora_max_rank * 2)
        max_num = common_config.heterlora_max_rank
        min_num = common_config.heterlora_min_rank
        powers_of_two = []  
        max_power = int(math.log2(max_num)) + 1
        for i in range(max_power):
            n = 2 ** (i+1) 
            if n >= min_num and n <= max_num:
                powers_of_two.append(n)
    else:
        raise NotImplementedError
    
    trainable_paras = 0
    for layer, para in global_model.named_parameters():
        if para.requires_grad:
            trainable_paras += para.numel()

        if "out_layer" in layer:
            trainable_paras += para.numel()
    trainable_paras_size = trainable_paras * 4 / 1024 / 1024
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
    partitial_data = args.partitial_data
    logger.info(f"totoal dataset: {partitial_data * 100}% {common_config.dataset_type}")
    if common_config.dataset_type in [ "cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2",  "stsb", "wnli", "ag_news", "20news"]:
        from mydatasets import get_glue_dataset
        train_dataset = get_glue_dataset(common_config.dataset_type, pretrained_model_path, "train", batch_size=common_config.batch_size)
        if common_config.dataset_type in ["ag_news", "20news"]:
            test_dataset = get_glue_dataset(common_config.dataset_type, pretrained_model_path, "test", batch_size=common_config.batch_size)
        else:
            test_dataset = get_glue_dataset(common_config.dataset_type, pretrained_model_path, "validation", batch_size=common_config.batch_size)
        
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(int(partitial_data * len(train_dataset))))
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

    client_num = args.client_num

    if args.data_pattern != 0:
        logger.info("non-IID partition prepare... ")
        logger.info(f"alpha --> {alpha}")
        if args.dataset_type == "sst2":    
            label_vocab = {'negative': 0, 'positive': 1}
            label_map = {0 : 'negative', 1 : 'positive'}
        
        elif args.dataset_type == "qnli" or args.dataset_type == "rte" or args.dataset_type == "wnli":    
            label_vocab = {'entailment': 0, 'not_entailment': 1}
            label_map = {0 : 'entailment', 1 : 'not_entailment'}

        elif args.dataset_type == "mrpc":    
            label_vocab = {'not_equivalent': 0, 'equivalent': 1}
            label_map = {0 : 'not_equivalent', 1 : 'equivalent'}

        elif args.dataset_type == "qqp":    
            label_vocab = {'not_duplicate': 0, 'duplicate': 1}
            label_map = {0 : 'not_duplicate', 1 : 'duplicate'}
        
        elif args.dataset_type == "mnli" or args.dataset_type == "mnli-mm":    
            label_vocab = {'entailment': 0, 'neutral': 1, "contradiction": 2}
            label_map = {0 : 'entailment', 1 : 'neutral', 2: "contradiction" }

        elif args.dataset_type == "cola":    
            label_vocab = {'unacceptable': 0, 'acceptable': 1}
            label_map = {0 : 'unacceptable', 1 : 'acceptable'}

        elif args.dataset_type == "ag_news":    
            label_vocab = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}
            label_map = {0 : 'World', 1 : 'Sports', 2 : 'Business', 3 : 'Sci/Tech'}

        elif args.dataset_type == "20news":
            label_vocab = {
                            'alt.atheism': 0,
                            'comp.graphics': 1,
                            'comp.os.ms-windows.misc': 2,  
                            'comp.sys.ibm.pc.hardware': 3,
                            'comp.sys.mac.hardware': 4,
                            'comp.windows.x': 5,
                            'misc.forsale': 6,
                            'rec.autos': 7, 
                            'rec.motorcycles': 8,
                            'rec.sport.baseball': 9,
                            'rec.sport.hockey': 10,
                            'sci.crypt': 11,
                            'sci.electronics': 12,
                            'sci.med': 13,
                            'sci.space': 14,
                            'soc.religion.christian': 15,
                            'talk.politics.guns': 16,  
                            'talk.politics.mideast': 17,
                            'talk.politics.misc': 18,
                            'talk.religion.misc': 19
                            }
            label_map = {
                        0: 'alt.atheism',
                        1: 'comp.graphics',
                        2: 'comp.os.ms-windows.misc',
                        3: 'comp.sys.ibm.pc.hardware',  
                        4: 'comp.sys.mac.hardware',
                        5: 'comp.windows.x',
                        6: 'misc.forsale',
                        7: 'rec.autos',
                        8: 'rec.motorcycles',
                        9: 'rec.sport.baseball',
                        10: 'rec.sport.hockey',
                        11: 'sci.crypt',
                        12: 'sci.electronics',
                        13: 'sci.med',
                        14: 'sci.space',
                        15: 'soc.religion.christian',
                        16: 'talk.politics.guns',
                        17: 'talk.politics.mideast',  
                        18: 'talk.politics.misc',
                        19: 'talk.religion.misc'
                        }
        
        else:
            raise NotImplementedError
        
        label_assignment_train = np.array([
            label_map[int(train_dataset[idx]['label'])]
            for idx in range(len(train_dataset))
        ])
        train_data_partition = label_skew_process(label_vocab, label_assignment_train, client_num, alpha, len(train_dataset), logger)
        train_data_partition = [[int(train_data_partition[x][y]) for y in range(len(train_data_partition[x]))] for x in range(len(train_data_partition))]
    else:
        logger.info("IID partition...")
        train_data_partition = RandomPartitioner(data_len=len(train_dataset), partition_sizes=[1/client_num for _ in range(client_num)])
    
    # train_data_partition = partition_data(common_config.dataset_type, common_config.data_pattern)

    # memory heterogeneity
    # memory_size = [4, 6, 8] 
    # memory_prop = [0.4, 0.3, 0.3]  

    # client_memory = np.random.choice(
    #     memory_size,
    #     size=worker_num, 
    #     p=memory_prop  
    # )
    edge = client_num // 3 + (1 if client_num % 3 != 0 else 0)
    nx = client_num // 3
    agx = client_num // 3
    logger.info(f"edge num --> {edge}, nx num --> {nx}, agx num --> {agx}")
    
    all_client_memory = [4 for _ in range(edge)] + [6 for _ in range(nx)] + [8 for _ in range(agx)]
    all_client_idxes = [i for i in range(len(all_client_memory))]
    logger.info(f"all_client_idxes --> {all_client_idxes}")
    clients_sample_count = [0 for i in range(len(all_client_idxes))]
    random.shuffle(all_client_memory)
    

    mu = 1 
    min_uploading_bandwidth = 1
    max_uploading_bandwidth = 3
    sigma = 2.25
    
    downloading_bandwidth = 15.0
    global_comm_cost = 0
    global_time = 0.0

    # 第一轮下发模型
    global_comm_cost += model_size * worker_num   
    global_time += model_size / downloading_bandwidth 

    max_acc = 0.0
    for epoch_idx in range(1, 1+common_config.epoch):
        logger.info(f"################## Round {epoch_idx} begin #####################")

        # 1. 采样对应worker数量的clients，下发配置和模型参数（内存、训练数据idx）
        sampled_clients_idxs = random.sample(all_client_idxes, worker_num)
        logger.info(f"sampled_clients_idxs --> {sampled_clients_idxs}")
        for idx in sampled_clients_idxs:
            clients_sample_count[idx] += 1

        for worker_idx, worker in enumerate(worker_list):
            worker.config.para = global_model.state_dict()

            # sample related
            worker.config.client_idx = sampled_clients_idxs[worker_idx]
            # logger.info(f"worker {worker_idx} has client {worker.config.client_idx}")
            if args.data_pattern != 0:
                worker.config.train_data_idxes = train_data_partition[worker.config.client_idx]
            else:
                worker.config.train_data_idxes = train_data_partition.use(worker.config.client_idx)
            
            
            # logger.info(f"worker {worker_idx} training idxes: {worker.config.train_data_idxes}")
            worker.config.source_train_dataset = train_dataset
            worker.config.test_dataset = test_dataset
            worker.config.memory = all_client_memory[worker.config.client_idx]
            if common_config.finetune_type == "heterlora":
                worker.config.heterlora_client_rank = random.choice(powers_of_two)

            # 根据设备的类型和方法，设置本地训练时间
            
            if "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "sst2":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.02
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.09
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.76
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.43
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.78
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.59
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.38
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.79
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.59
                elif common_config.finetune_type == "our_avg":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.38
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.79
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.59
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.02
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.09
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.76

            elif "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "qnli":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.99
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.54
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.38
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.67
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.38
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.27
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.69
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.39
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.30
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.99
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.54
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.38

            elif "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "rte":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.34
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.73
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.5
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.93
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.53
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.39
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.92
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.53
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.4
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.34
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.73
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.5
            elif "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "mrpc":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.95
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.57
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.39
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.68
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.42
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.29
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.67
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.4
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.3
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.95
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.57
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.39
            elif "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "cola":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.22
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.68
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.47
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.84
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.49
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.34
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.85
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.49
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.37
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.22
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.68
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.47
            elif "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "qqp":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.81
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.41
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.28
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.46
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.31
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.21
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.45
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.29
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.21
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 0.81
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.41
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.28
            elif "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "mnli":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.09
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.1
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.74
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.38
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.64
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.42
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.24
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.66
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.45
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.09
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.1
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.74
            elif "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "ag_news":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 3.42
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.84
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 1.18
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.47
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.27
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.85
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.33
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.31
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.86
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 3.42
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.84
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 1.18
            elif "bert-base-uncased" in pretrained_model_path and common_config.dataset_type == "20news":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 3.17
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.46
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.95
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.92
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.02
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.72
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.85
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.04
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.71
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 3.17
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.46
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.95

            elif "roberta-base" in pretrained_model_path and common_config.dataset_type == "sst2":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.94
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.21
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.76
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.37
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.77
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.53
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.35
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 0.78
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.52
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.94
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.21
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.76

            elif "roberta-base" in pretrained_model_path and common_config.dataset_type == "ag_news":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 3.13
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.59
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 1.1
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.06
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.16
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.76
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.02
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.17
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.79
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 3.13
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.59
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 1.1
            
            elif "roberta-base" in pretrained_model_path and common_config.dataset_type == "20news":
                if common_config.finetune_type == "fedft":
                    pass
                elif common_config.finetune_type == "fedlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.71
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.48
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.94
                elif common_config.finetune_type == "fedadapter":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.82
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.02
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.7
                elif common_config.finetune_type == "our":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 1.83
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.02
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.7
                elif common_config.finetune_type == "our_avg":
                    raise NotImplementedError
                elif common_config.finetune_type == "heterlora":
                    if worker.config.memory == 4:
                        worker.config.local_training_time = 2.71
                    elif worker.config.memory == 6:
                        worker.config.local_training_time = 1.48
                    elif worker.config.memory == 8:
                        worker.config.local_training_time = 0.94

            else:
                raise NotImplementedError
            # logger.info(f"$$$$$$$$$$$ worker {worker_idx} --> {worker.config.train_data_idxes}")
        
        ###################### Sending init config to clients ###############################
        logger.info(f"Sending config to all workers and start the training procedure")
        communication_parallel(worker_list, 1, comm, action="init")



        logger.info("Waiting and receiving updated paras from clients")
        communication_parallel(worker_list, epoch_idx, comm, action="get_para")
        # 根据接收的参数，计算客户端发送的可训练参数的时间和服务器下发模型的时间，根据设备的类型得到本地计算时间
        ## 每一轮的时间都是由本地训练时间和通讯时间之和中最大的值 max(local_training + uploading_time)
        current_round_time = 0
        with torch.no_grad():
            for worker_idx in range(len(worker_list)):
                clinet_uploaded_para_size = 0
                for layer, paras in worker_list[worker_idx].config.neighbor_paras.items():
                    clinet_uploaded_para_size += paras.numel()
                clinet_uploaded_para_size = clinet_uploaded_para_size * 4 / 1024 / 1024
                global_comm_cost += clinet_uploaded_para_size # 上传服务器通信量
                uploading_bandwidth = np.random.normal(mu, sigma, 1)
                uploading_bandwidth = np.clip(uploading_bandwidth, min_uploading_bandwidth, max_uploading_bandwidth)[0]
                client_uploading_time = clinet_uploaded_para_size / uploading_bandwidth
                client_local_training_time = worker_list[worker_idx].config.local_training_time * (len(worker.config.train_data_idxes) / common_config.batch_size)
                current_round_time = max(current_round_time, client_uploading_time + client_local_training_time)
        logger.info(f"current round time (local training + uploading) --> {current_round_time}")
        ## 模型上传和训练时间的最大值，加上下发模型参数的时间
        logger.info(f"current distribute time --> {trainable_paras_size / downloading_bandwidth}")
        global_time += current_round_time + trainable_paras_size / downloading_bandwidth
        global_comm_cost += trainable_paras_size * worker_num
        logger.info(f"Current global time --> {global_time}")
        logger.info(f"Current global comm cost --> {global_comm_cost}")

        logger.info("Clients' information received.")
        logger.info("Performing aggregation...")
        global_para = parameter_wise_aggregation(worker_list, common_config)
        # logger.info("Aggregation finished and sending the newly aggregated paras back to clients")
        # communication_parallel(worker_list, epoch_idx, comm, action="send_model",data=global_para)
        logger.info(f"TEST on server...")
        # logger.info("The aggregated model is: ")
        # for layer, paras in global_model.named_parameters():
        #     logger.info(f"\t{layer} --> {paras}")
        global_model_sd = global_model.state_dict()
        global_model_sd.update(global_para)
        global_model.load_state_dict(global_model_sd)
        global_model.to("cuda:0")
        global_model.eval()
        
        if common_config.dataset_type in [ "cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2",  "stsb", "wnli", "ag_news", "20news"]:
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
            max_acc = max(max_acc, mean(metric_all))
            if global_model.metric_1 is not None:
                logger.info(f"test {metric_1_name} -->  {mean(metric_1_all)}")

        logger.info(f"Round {epoch_idx} finished")
    
    logger.info(f"max_acc --> {max_acc}")

    # close socket
    
def parameter_wise_aggregation(worker_list, common_config: CommonConfig):
    overall_para = dict()
    lora_selected_clients_num = 10
    with torch.no_grad():
        logger.info("aggregate paras at server side...")
        for worker_idx in range(len(worker_list)):
            # logger.info(f"\t From client {worker_idx}")
            for layer, paras in worker_list[worker_idx].config.neighbor_paras.items():
                # logger.info(f"\t\t {layer} --> {paras}")
                if layer not in overall_para:
                    overall_para[layer] = dict()
                    overall_para[layer]['cnt'] = 1
                    overall_para[layer]['value'] = paras
                else:
                    if common_config.finetune_type == "our":
                        if overall_para[layer]['cnt'] <= lora_selected_clients_num:
                            overall_para[layer]['cnt'] += 1
                            overall_para[layer]['value'] += paras
                        else:
                            continue
                    else:
                        overall_para[layer]['cnt'] += 1
                        overall_para[layer]['value'] += paras
    aggregate_para_dict = dict()
    for layer in overall_para.keys():
        # logger.info(f"{layer} --> {overall_para[layer]['cnt']} --> {overall_para[layer]['value']}")
        aggregate_para_dict[layer] = overall_para[layer]['value'] / overall_para[layer]['cnt']
        # logger.info(f"{layer} after aggregated --> {aggregate_para_dict[layer]}")
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
