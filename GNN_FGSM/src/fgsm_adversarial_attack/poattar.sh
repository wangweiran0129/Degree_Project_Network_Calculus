#!/bin/bash
#SBATCH --chdir /home/weirwang/Degree_Project_Network_Calculus/DeepFP_gnn-main/src/analysis/

#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 2
#SBATCH --mem 48G
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --time 06:00:00

echo Running on `hostname`

source /home/weirwang/venvs/izar-python/bin/activate

python -m potential_attack_target "../../../Network_Information_and_Analysis/prediction_value/"
