#!/bin/bash


# 同时开启系统和数据异构，测试集中后面的（our），以及平均之间的差距
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type our --our_total_rank 192 --lr 2e-2                            --partitial_data 0.1 --data_pattern 1  --alpha 0.5 --enable_sys_heter True : -n 20 python client.py
mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type our_avg --our_total_rank 192 --lr 2e-2                            --partitial_data 0.1 --data_pattern 1  --alpha 0.5 --enable_sys_heter True : -n 20 python client.py

