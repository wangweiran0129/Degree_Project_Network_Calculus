#!/bin/bash
#SBATCH --chdir /home/weirwang/Degree_Project_Network_Calculus/DeepFP_gnn-main/src/model/

#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --mem 40G
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --time 06:00:00

echo Running on `hostname`

source /home/weirwang/venvs/izar-python/bin/activate

python -m train_model "/home/weirwang/serialized_dataset/train_graphs_pmoo.pickle" "/home/weirwang/serialized_dataset/train_targets_pmoo.pickle" "/home/weirwang/serialized_dataset/test_graphs_pmoo.pickle" "/home/weirwang/serialized_dataset/test_targets_pmoo.pickle" 1.4*5e-4 30
