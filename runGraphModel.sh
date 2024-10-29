#!/bin/bash
#SBATCH --partition=M1
#SBATCH --qos=q_d8_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --job-name=runGCNs
#SBATCH --output=./output/output_%j.out 
#SBATCH --error=./error/error_%j.err
#SBATCH --time=0-06:00:00


module load anaconda
eval "$(conda shell.bash hook)"
conda activate base


echo "Running GCNs"
python ~/MscProjectDataAnalysis/src/runGraphmodels.py --model GCN --num_epochs 10 --learning_rate 0.001 --batch_size 8

# echo "Running GATs"
# python ~/MscProjectDataAnalysis/src/runGraphModels.py --model GAT--num_epochs 10 --learning_rate 0.001 --batch_size 16

# echo "Running GATWithDropout"
# python ~/MscProjectDataAnalysis/src/runGraphModels.py --model GATWithDropout --num_epochs 10 --learning_rate 0.001 --batch_size 16

# echo "Running GATWithBatchNorm"
# python ~/MscProjectDataAnalysis/src/runGraphModels.py --model GATWithBatchNorm --num_epochs 10 --learning_rate 0.001 --batch_size 16

# echo "Running ModifiedGCN"
# python ~/MscProjectDataAnalysis/src/runGraphModels.py --model ModifiedGCN --num_epochs 10 --learning_rate 0.001 --batch_size 16

