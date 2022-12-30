#!/bin/bash
#SBATCH --chdir /home/weirwang/Degree_Project_Network_Calculus/DeepFP_gnn-main/src/data/large_network_generation/

#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 2
#SBATCH --mem 48G
#SBATCH --time 24:00:00

echo Running on `hostname`

source /home/weirwang/venvs/fidis-adversarial-attack-gnn/bin/activate

java -jar NetCal.jar &
MY_JAVA_PID=$!
python -m large_network_generation_pbz 0 100
kill $MY_JAVA_PID
exit 0