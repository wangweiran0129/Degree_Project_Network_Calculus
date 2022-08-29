#!/bin/bash
#SBATCH --chdir /home/weirwang/Degree_Project_Network_Calculus/DeepFP_gnn-main/src/data/large_network_generation/

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

java -jar NetCal.jar &
MY_JAVA_PID=$!
python -m large_network_generation_pbz 200 "../../model/deepfpPMOO.pt"
kill $MY_JAVA_PID
exit 0