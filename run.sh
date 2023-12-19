#!/bin/bash

# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 2 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 4 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 8 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 16 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 32 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 64 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 128 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 256 : -n 10 python client.py 
# mpiexec -n 2 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type our --fedlora_rank 2 --lr 2e-5 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedft --fedlora_rank 2 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedlora --fedlora_rank 16 : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 80 --finetune_type fedadapter --fedlora_rank 2 : -n 20 python client.py 

# TODO
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type our --fedlora_rank 2 --lr 2e-5 --our_total_rank 192 --data_pattern 1 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type our --fedlora_rank 2 --lr 2e-5 --our_total_rank 384 --data_pattern 1 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type our --fedlora_rank 2 --lr 1e-5 --our_total_rank 768 --data_pattern 1 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type our --fedlora_rank 2 --lr 2e-5 --our_total_rank 1536 --data_pattern 1 : -n 20 python client.py
 

# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type fedlora --fedlora_rank 16 --lr 2e-5 --data_pattern 1 : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type fedlora --fedlora_rank 32 --lr 2e-5 --data_pattern 1 : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type fedlora --fedlora_rank 64 --lr 2e-5 --data_pattern 1 : -n 20 python client.py

# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type fedadapter --fedadpter_width 32 --fedadpter_depth 12 --lr 2e-5 --data_pattern 1 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type fedadapter --fedadpter_width 64 --fedadpter_depth 6 --lr 2e-5 --data_pattern 1 : -n 20 python client.py 

# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type heterlora --lr 2e-5 --data_pattern 1 : -n 20 python client.py 

# 12.18
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 100 --finetune_type our --fedlora_rank 2 --lr 1e-2 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 100 --finetune_type our --fedlora_rank 2 --lr 2e-2 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 100 --finetune_type our --fedlora_rank 2 --lr 3e-2 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 100 --finetune_type our --fedlora_rank 2 --lr 4e-2 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 100 --finetune_type our --fedlora_rank 2 --lr 5e-2 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 

# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 100 --finetune_type our --fedlora_rank 2 --lr 5e-3 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 100 --finetune_type our --fedlora_rank 2 --lr 3e-3 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 100 --finetune_type our --fedlora_rank 2 --lr 1e-3 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 

# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 200 --finetune_type fedft --fedlora_rank 2 --lr 1e-2 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 

mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 200 --finetune_type fedlora --fedlora_rank 2 --lr 1e-2 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 

mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 200 --finetune_type fedadapter --fedlora_rank 2 --lr 1e-2 --our_total_rank 192 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 


# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type our --fedlora_rank 2 --lr 2e-5 --our_total_rank 1536 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type our --fedlora_rank 2 --lr 1e-4 --our_total_rank 768 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type our --fedlora_rank 2 --lr 1e-4 --our_total_rank 1536 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py

# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type fedft --fedlora_rank 2 --data_pattern 0 --partitial_data 0.02 : -n 20 python client.py 
# mpiexec -n 1 python server.py --dataset_type sst2 --batch_size 32 --epoch 30 --finetune_type fedft --fedlora_rank 2 --data_pattern 1 --partitial_data 0.02 : -n 20 python client.py 