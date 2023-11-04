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

# TODO fix BertForQA model
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

parser = argparse.ArgumentParser(description='Distributed Client')
parser.add_argument('--visible_cuda', type=str, default='-1')
parser.add_argument('--use_cuda', action="store_false", default=True)

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

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

def set_seed(config):
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

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
    logger.info("Acknowledged.")
    
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
    # init config
    config = BertForMRCConfig()
    
    config.model_dir = "/data/jliu/models"
    config.train_path = "/data/jliu/data/SQuAD"
    config.dev_path = "/data/jliu/data/SQuAD"
    config.device = f"cuda:{rank % torch.cuda.device_count()}"
    set_seed(config)
    logger.info(f"Loading pretrained model from {os.path.join(config.model_dir,config.model_name)}...")
    tokenizer = BertTokenizer.from_pretrained(os.path.join(config.model_dir,config.model_name))
    model = BertForQA(config)
    
    ft_type = "FT"
    if ft_type == "FT":
        pass
    elif ft_type == "FLoRA_QKV":
        target_modules = ["query", "key", "value"]
        logger.info("Finished.")
        
        lora_config = LoraConfig(
            r = 8,
            lora_alpha = 32,
            target_modules=target_modules,
            lora_dropout = 0.05,
            bias = "none",
            task_type = "QA",
        )
        logger.info(f"lora config --> {lora_config}")
        
        
        # model = prepare_model_for_int8_training(model)

        model = get_peft_model(model, lora_config)
    elif ft_type == "PLoRA_QKV":
        import loralib as lora
        target_attn_matrix = { # attn
            "9": ["query", "key", "value"],
            "10": ["query", "key", "value"],
            "11": ["query", "key", "value"]
        }
        target_ffn_matrix = { # ffn
            "11": ["intermediate", "output"]
        }
        for layer in target_attn_matrix.keys():
            for matrix in target_attn_matrix[layer]:
                module = model._modules["BertModule"]._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix]
                lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=8, lora_alpha=32)
                lora_layer.weight = module.weight
                lora_layer.bias = module.bias
                model._modules["BertModule"]._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules["attention"]._modules["self"]._modules[matrix] = lora_layer
            for layer in target_ffn_matrix.keys():
                for matrix in target_ffn_matrix[layer]:
                    module = model._modules["BertModule"]._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"]
                    lora_layer = lora.Linear(in_features=module.in_features, out_features=module.out_features, r=8, lora_alpha=32)
                    lora_layer.weight = module.weight
                    lora_layer.bias = module.bias
                    model._modules["BertModule"]._modules["bert"]._modules["encoder"]._modules["layer"]._modules[layer]._modules[matrix]._modules["dense"] = lora_layer
            lora.mark_only_lora_as_trainable(model)
    
    model.to(config.device)
    
    logger.info(f"Fine-Tuning type = {ft_type}")
    # logger.info(f"The size of trainable parameters of the peft model is {model.get_nb_trainable_parameters()}")
    logger.info(f"The model architecture --> ")
    for layer, para in model.named_parameters():
        logger.info(layer)
    
    logger.info(f"Loading dataset from {os.path.join(config.train_path, config.train_file)}")
    train_Dataset = SQuAD_V2_Dataset(tokenizer=tokenizer,data_dir=config.train_path,filename=config.train_file,is_training=True,config=config,cached_features_file=os.path.join(config.train_path,"cache_" + config.train_file.replace("json","data")))
    train_features,train_dataset = train_Dataset.features,train_Dataset.dataset
    logger.info("Finished.")

    train_loader = mydatasets.create_dataloaders(dataset=train_dataset, batch_size=config.batch_size, selected_idxs=client_config.train_data_idxes)

    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(client_config.train_data_idxes))
    logger.info("  Num local steps = %d", config.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", config.batch_size)
    

    while True:
        # 开始本地训练
        logger.info(f"##################### Round {common_config.tag} start ... #########################")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        tasks.append(
            asyncio.ensure_future(
                # train_and_eval(config,model,train_loader,tokenizer, rank, logger)
                local_procedure(comm, common_config, config, model, train_loader, tokenizer, rank, logger)
            )
        )
        loop.run_until_complete(asyncio.wait(tasks))

        loop.close()
  
        common_config.tag += 1
        if common_config.tag==common_config.epoch+1:
            break

async def local_procedure(comm, common_config, config:BertForMRCConfig, model:PeftModel, train_loader, tokenizer, rank, logger):
    train_and_eval(config,model,train_loader,tokenizer, rank, logger)
    import time
    time.sleep(10)
    # 本地训练结束，上传lora参数
    logger.info("Sending local parameters to the server")
    local_paras = dict()
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
