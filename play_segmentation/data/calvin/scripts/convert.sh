#! /bin/bash

#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH -c 18
#SBATCH -o JOB%j.out # File to which STDOUT will be written
#SBATCH -e JOB%j.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nhopner@gmail.com

source /projects/0/prjs1044/miniconda3/bin/activate

conda activate play_segmentation

python convert.py --config ./configs/convert.yaml
