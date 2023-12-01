#!/bin/bash

mpiexec -n 1 python server.py : -n 8 python client.py 
# mpiexec -n 2 python client.py
