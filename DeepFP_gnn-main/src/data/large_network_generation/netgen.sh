#!/bin/bash
#SBATCH --chdir /home/weirwang/Degree_Project_Network_Calculus/DeepFP_gnn-main/src/data/large_network_generation/

#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 2
#SBATCH --mem 4G
#SBATCH --time 72:00:00

echo "fidis $HOSTNAME"

module load gcc python

source /home/weirwang/venvs/fidis-adversarial-attack-gnn/bin/activate

java -jar NetCal.jar &
MY_JAVA_PID=$!
python large_network_generation_pbz.py
kill $MY_JAVA_PID
exit 0