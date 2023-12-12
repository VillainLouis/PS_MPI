#!/bin/bash

if [ $# -lt 1]; then
    echo "A new name is need for current result."
    echo "Usage: mv_results.sh [new_record_name]"
    exit 1
fi

result_name=$1
result_path="./Results"
if [ ! -d "$result_path" ]; then
    mkdir "$result_path"
    echo "mkdir $result_path"
fi

folder_name="$result_path/$result_name"
if [ ! -d "$folder_name" ]; then
    mkdir "$folder_name"
    echo "mkdir $folder_name"
    mv clients "$folder_name/clients"
    mv server "$folder_name/server"
    # mv output "$folder_name/output"
    echo "mv result finished"
else
    echo "$file_name already exists."
fi
