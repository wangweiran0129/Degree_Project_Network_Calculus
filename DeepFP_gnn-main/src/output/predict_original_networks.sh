#!/bin/bash
#SBATCH --chdir /home/weirwang/Degree_Project_Network_Calculus/DeepFP_gnn-main/src/output/

#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --mem 48G
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --time 06:00:00

echo Running on `hostname`

source /home/weirwang/venvs/izar-python/bin/activate

java -jar NetCal.jar &
MY_JAVA_PID=$!
python -m predict_original_networks "../model/deepfpPMOO.pt" "../data/large_network_generation/dataset-attack-large.pbz"
kill $MY_JAVA_PID
exit 0