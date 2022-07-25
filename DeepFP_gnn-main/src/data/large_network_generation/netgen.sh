#!/bin/bash
#SBATCH --chdir /home/<epfl_user_id>/<dir_path>/

#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 2
#SBATCH --mem 4G
#SBATCH --time 24:00:00

echo "fidis $HOSTNAME"

module load gcc python

source /home/<epfl_user_id>/venvs/<virtual_environment_name>/bin/activate

java -jar NetCal.jar &
MY_JAVA_PID=$!
python large_network_generation_pbz.py
kill $MY_JAVA_PID
exit 0