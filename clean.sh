#!/bin/bash

if test -d ./server; then
    echo "rm -rf server && clients"
    rm -rf server clients
else 
    echo "no record to be cleaned"
fi