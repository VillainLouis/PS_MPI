#!/bin/bash

mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 2 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 4 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 8 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 16 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 32 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 64 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 128 : -n 10 python client.py 
# mpiexec -n 1 python server.py --epoch 5 --finetune_type fedlora --fedlora_rank 256 : -n 10 python client.py 
# mpiexec -n 2 python client.py
