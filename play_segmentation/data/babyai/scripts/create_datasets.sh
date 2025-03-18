#! /bin/bash

python generate.py --config ./config/gen_data.yaml --n_trajectories 10000
python generate.py --config ./config/gen_data.yaml --n_trajectories 25000
python generate.py --config ./config/gen_data.yaml --n_trajectories 50000
python generate.py --config ./config/gen_data.yaml --n_trajectories 100000
