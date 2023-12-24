#!/bin/bash

# 开始测试系统和数据异构下各方法的对比，明天完成时间记录和流量记录
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedlora --fedlora_rank 16 --fedlora_depth 12 --lr 2e-2        --partitial_data 1.0 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedadapter --fedadpter_width 32 --fedadpter_depth 6 --lr 2e-3 --partitial_data 1.0 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type our --our_total_rank 192 --lr 2e-2                            --partitial_data 1.0 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type heterlora --max_rank 64 --min_rank 2 --lr 2e-2                --partitial_data 1.0 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py

# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedlora --fedlora_rank 16 --fedlora_depth 12 --lr 2e-2        --partitial_data 0.1 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedadapter --fedadpter_width 32 --fedadpter_depth 6 --lr 2e-3 --partitial_data 0.1 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type our --our_total_rank 192 --lr 2e-2                            --partitial_data 0.1 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type heterlora --max_rank 64 --min_rank 2 --lr 2e-2                --partitial_data 0.1 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py

# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedlora --fedlora_rank 16 --fedlora_depth 12 --lr 2e-2        --partitial_data 0.02 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedadapter --fedadpter_width 32 --fedadpter_depth 6 --lr 2e-3 --partitial_data 0.02 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type our --our_total_rank 192 --lr 2e-2                            --partitial_data 0.02 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type heterlora --max_rank 64 --min_rank 2 --lr 2e-2                --partitial_data 0.02 --data_pattern 1  --alpha 1.0 --enable_sys_heter True : -n 20 python client.py