#!/bin/bash

mpiexec -n 1 python server.py --batch_size 8 : -n 8 python client.py --fune_type "FT" --local_step 200
# mpiexec -n 2 python client.py
