#!/bin/bash

mpiexec -n 1 python server.py : -n 10 python client.py
# mpiexec -n 2 python client.py