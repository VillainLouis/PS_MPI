from comm_utils import *
import torch
from transformers import BertTokenizer

class ClientAction:
    LOCAL_TRAINING = "local_training"


class Worker:
    def __init__(self, config, rank):
        #这个config就是后面的client_config
        self.config = config
        self.rank = rank

    async def send_data(self, data, comm, epoch_idx):
        await send_data(comm, data, self.rank, epoch_idx)    

    async def send_init_config(self, comm, epoch_idx):
        print(f"sending init config to {self.rank}")
        await send_data(comm, self.config, self.rank, epoch_idx)    

    async def get_model(self, comm, epoch_idx):
        self.config.neighbor_paras = await get_data(comm, self.rank, epoch_idx)
    
    async def get_para(self, comm, epoch_idx):
        train_time,send_time = await get_data(comm, self.rank, epoch_idx)
        self.config.train_time=train_time
        self.config.send_time=send_time

class CommonConfig:
    def __init__(self):
        self.model_type = None
        self.dataset_type = None
        self.batch_size = None
        self.data_pattern = None
        self.lr = None
        self.decay_rate = None
        self.min_lr = None
        self.epoch = None
        self.momentum=None
        self.weight_decay=None
        self.para = None
        self.data_path = None
        #这里用来存worker的
        # self.bert_mrc_config = BertForMRCConfig()
        # self.lora_layers = []
        self.finetune_type = None
        self.fedlora_rank = None
        self.fedlora_depth = None

        self.heterlora_max_rank = None
        self.heterlora_min_rank = None
        self.client_rank = None

        self.our_total_rank = None
        self.fedadpter_width = None
        self.fedadpter_depth = None

        self.enable_sys_heter = None

        self.test_target_matrix = None

class ClientConfig:
    def __init__(self,
                common_config,
                custom: dict = dict()
                ):
        # client 自己的模型参数
        self.para = None
        # client 自己的dataset
        self.train_data_idxes = None
        self.source_train_dataset = None
        self.test_dataset = None

        self.common_config=common_config

        self.average_weight=0.1
        self.local_steps=20
        self.compre_ratio=1
        self.train_time=0
        self.send_time=0
        self.neighbor_paras=None
        self.neighbor_indices=None

        # resource
        self.memory=None

        # heterlora
        self.heterlora_client_rank = None

        self.local_training_time = None

        self.client_idx = None


class BertForMRCConfig(object):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_gpu = torch.cuda.device_count()
        self.seed = 18

        self.model_name = "bert-base-uncased"
        self.max_seq_length = 512
        self.doc_stride = 128
        self.max_query_length = 64
        self.Tokenizer = BertTokenizer
        self.num_type = 2

        self.train_path = "data/train"
        self.train_file = "train-v2.0.json"
        self.dev_path = "data/dev"
        self.dev_file = "dev-v2.0.json"
        self.check_point_path = "" # 是否继续训练

        self.model_dir = "/data/jliu/models/"
        self.logging_path = "all.log"
        self.output_dir = "output/"
        self.data_output_dir = ""


        self.batch_size = 8
        self.learning_rate = 1e-4
        # self.optimizer = 'Adam'
        self.adam_epsilon = 1e-8
        self.nums_epochs = 1
        self.max_steps = 200 # 200 800 1000 # 1000000000
        self.save_steps = 10000
        self.gradient_accumulation_steps = 10
        self.logging_steps = 10
        self.warmup_steps = 0


        self.hidden_size = 768
        self.num_class = 10

        self.n_best_size = 20 # "The total number of n-best predictions to generate in the nbest_predictions.json output file."
        self.max_answer_length = 30 # "The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another."
        self.do_lower_case = True
        self.verbose_logging = True
        self.version_2_with_negative = True
        self.null_score_diff_threshold = 0.0 # "If null_score - best_non_null is greater than the threshold predict null."