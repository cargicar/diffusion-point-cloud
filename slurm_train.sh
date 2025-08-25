#!/bin/bash

#SBATCH --job-name=omnilearn_shapenet
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --qos=regular
#SBATCH --account=m3246
##SBATCH --volume="/pscratch/sd/c/ccardona:/pscratch/sd/c/ccardona"
##SBATCH  --image=docker:vmikuni/pytorch:ngc-23.12-v0

##srun  shifter python scripts/train_jetnet.py --local --layer_scale --dataset jetnet30 --fine_tune 
module load conda
conda activate dpm-pc-gen
module load pytorch
srun python train_gen.py 