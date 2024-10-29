#!/bin/bash
#SBATCH --partition=M1
#SBATCH --qos=q_d8_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --job-name=runNNs
#SBATCH --output=./output/output_%j.out 
#SBATCH --error=./error/error_%j.err
#SBATCH --time=0-06:00:00


module load anaconda
eval "$(conda shell.bash hook)"
conda activate base

echo "Running BaselineNN"
python ~/MscProjectDataAnalysis/src/runNeuralNetworks.py --model BaselineNN --num_epochs 100 --learning_rate 0.001 --batch_size 32 --checkpoint_dir checkpoints_final_NNModels --loss_function BCE

echo "Running DropoutNN"
python ~/MscProjectDataAnalysis/src/runNeuralNetworks.py --model DropoutNN --num_epochs 100 --learning_rate 0.001 --batch_size 32 --checkpoint_dir checkpoints_final_NNModels --loss_function BCE

echo "Running SkipConnectionNN"
python ~/MscProjectDataAnalysis/src/runNeuralNetworks.py --model SkipConnectionNN --num_epochs 100 --learning_rate 0.001 --batch_size 32 --checkpoint_dir checkpoints_final_NNModels --loss_function BCE

echo "Running DropoutSkipConnectionNN"
python ~/MscProjectDataAnalysis/src/runNeuralNetworks.py --model DropoutSkipConnectionNN --num_epochs 100 --learning_rate 0.001 --batch_size 32 --checkpoint_dir checkpoints_final_NNModels --loss_function BCE
