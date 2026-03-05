#!/bin/bash
#SBATCH --job-name=RHEA_print
#SBATCH --partition=savio2
#SBATCH --account=fc_minkwon
#SBATCH --qos=savio_normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=64G

#SBATCH --array=0-9
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err

# Debug
echo "Date                = $(date)"
echo "Hostname            = $(hostname -s)"
echo "Working Directory   = $(pwd)"
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
SECONDS=0

# Conda environment
source /global/scratch/users/minkwon/miniconda3/etc/profile.d/conda.sh
conda activate rhea-am

# Run from the project root directory (where step4_printability_map.py lives)
cd $SLURM_SUBMIT_DIR

# Set total batches (must match --array range count)
export TOTAL_BATCHES=10

python -u step4_printability_map.py

echo "-----------------------------------"
echo "Total execution time : $(($SECONDS / 60))m $(($SECONDS % 60))s"
echo "-----------------------------------"
