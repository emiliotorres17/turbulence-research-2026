#!/bin/bash

#SBATCH -p parallel
##SBATCH -p fn1
#SBATCH -n 1
##SBATCH --mem=95000
#SBATCH --qos=normal
#SBATCH -t 1-12:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eetorres@asu.edu
module purge
module load openmpi/3.0.0-intel-2017x
module load python/3.6.4

python3.6 ales_TF5_selective_limiting.py -f HIT_demo_inputs.txt
srun --mpi=pmi2 -n 1 python3.6 ales_TF5_selective_limiting.py -f HIT_demo_inputs.txt 
