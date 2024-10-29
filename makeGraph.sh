#!/bin/bash
#SBATCH --partition=M1
#SBATCH --qos=q_d8_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --job-name=makeGraph
#SBATCH --output=./output/output_%j.out 
#SBATCH --error=./error/error_%j.err
#SBATCH --time=0-06:00:00

module load anaconda
eval "$(conda shell.bash hook)"
conda activate base

# make network
python ~/MscProjectDataAnalysis/src/makeGraph.py 