#!/bin/bash

# 20

# r = 16, 2e-3效果不好, 换200client，lora 2e-2尝试ii

mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 1000 --finetune_type fedlora --fedlora_rank 16 --fedlora_depth 12 --lr 2e-3        --partitial_data 1.0 --data_pattern 1  --alpha 1.0  : -n 4 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 1000 --finetune_type fedadapter --fedadpter_width 32 --fedadpter_depth 6 --lr 2e-3 --partitial_data 1.0 --data_pattern 1  --alpha 1.0  : -n 10 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 1000 --finetune_type heterlora --max_rank 64 --min_rank 2 --lr 2e-3                --partitial_data 1.0 --data_pattern 1  --alpha 1.0  : -n 10 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 1000 --finetune_type our --our_total_rank 192 --lr 2e-3                            --partitial_data 1.0 --data_pattern 1  --alpha 1.0  : -n 10 python client.py


# 测试200client
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 200 --finetune_type fedlora --fedlora_rank 16 --fedlora_depth 12 --lr 2e-2        --partitial_data 1.0 --data_pattern 1  --alpha 10.0  : -n 10 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 200 --finetune_type fedadapter --fedadpter_width 32 --fedadpter_depth 6 --lr 2e-3 --partitial_data 1.0 --data_pattern 1  --alpha 10.0  : -n 10 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 200 --finetune_type heterlora --max_rank 64 --min_rank 2 --lr 2e-2                --partitial_data 1.0 --data_pattern 1  --alpha 10.0  : -n 10 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 200 --finetune_type our --our_total_rank 192 --lr 2e-2                            --partitial_data 1.0 --data_pattern 1  --alpha 10.0  : -n 10 python client.py

# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 200 --finetune_type fedlora --fedlora_rank 16 --fedlora_depth 12 --lr 2e-2        --partitial_data 1.0 --data_pattern 1  --alpha 1.0  : -n 10 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 200 --finetune_type fedadapter --fedadpter_width 32 --fedadpter_depth 6 --lr 2e-3 --partitial_data 1.0 --data_pattern 1  --alpha 1.0  : -n 10 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 200 --finetune_type heterlora --max_rank 64 --min_rank 2 --lr 2e-2                --partitial_data 1.0 --data_pattern 1  --alpha 1.0  : -n 10 python client.py
# mpiexec -n 1 python server.py --dataset_type ag_news --batch_size 8 --epoch 200 --client_num 200 --finetune_type our --our_total_rank 192 --lr 2e-2                            --partitial_data 1.0 --data_pattern 1  --alpha 1.0  : -n 10 python client.py


