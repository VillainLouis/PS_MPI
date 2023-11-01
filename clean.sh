#!/bin/bash

if test -d ./server; then
    echo "rm -rf server"
    rm -rf server
else 
    echo "no record to be cleaned"
fi

if test -d ./clients; then
    echo "rm -rf clients"
    rm -rf clients
else 
    echo "no record to be cleaned"
fi

if test -d ./output; then
    echo "rm -rf ./output/*"
    rm -rf output/*
else 
    echo "no record to be cleaned"
fi