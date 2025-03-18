#! /bin/bash

mkdir -p ../../../datasets/calvin

cd ../../../datasets/calvin


if [ -f "./calvin_debug_dataset.zip" ]; then
    echo "File already exists, no download needed"
else
    echo "Downloading debug dataset ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
fi

if [ -d "./calvin_debug_dataset" ]; then
    echo "Already unzipped, no need to unzip again"
else
    unzip calvin_debug_dataset.zip
    echo "saved folder: calvin_debug_dataset"
    rm calvin_debug_dataset.zip
fi



if [ -f "./task_D_D.zip" ]; then
    echo "File already exists, no download needed"
else
    echo "Downloading debug dataset ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
fi

if [ -d "./task_D_D" ]; then
    echo "Already unzipped, no need to unzip again"
else
    unzip task_D_D.zip
    echo "saved folder: task_D_D"
    rm task_D_D.zip
fi
