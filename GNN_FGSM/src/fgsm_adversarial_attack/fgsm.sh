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

java -jar NetCal.jar &
MY_JAVA_PID=$!
python -m adversarial_attack "../gnn_fp/model/deepfpPMOO.pt" "../../../Network_Information_and_Analysis/potential_attack_target2.csv" "../../../Network_Information_and_Analysis/original_topology/before_fp/pbz/" "../../../Network_Information_and_Analysis/original_topology/before_fp/pickle/"
kill $MY_JAVA_PID
exit 0
