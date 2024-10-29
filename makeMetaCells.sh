#!/bin/bash
#SBATCH --partition=M2
#SBATCH --qos=q_a_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --job-name=makeMetaCells
#SBATCH --output=./output/output_%j.out 
#SBATCH --error=./error/error_%j.err
#SBATCH --time=0-06:00:00

module load anaconda
eval "$(conda shell.bash hook)"
conda activate my_r_env

# Rscript ~/MscProjectDataAnalysis/src/makeMetaCells.r 

# Rscript ~/MscProjectDataAnalysis/src/getVariableGenes.r 

# Rscript ~/MscProjectDataAnalysis/src/mapToEnsembl.r 

Rscript ~/MscProjectDataAnalysis/src/makeH5adFiles.r 
