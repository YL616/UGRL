#!/bin/bash
#SBATCH -A test
#SBATCH -J ugrl
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH -p short
#SBATCH -t 3-0:00:00
#SBATCH -o ugrl.out

env=runner3.7

source activate $env

port=29501
crop_size=512

file=scripts/train_voc.py
config=configs/voc.yaml

work_dir=work_dir

echo python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir $work_dir
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$port $file --config $config --pooling gmp --crop_size $crop_size --work_dir $work_dir
