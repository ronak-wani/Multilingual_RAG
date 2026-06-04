#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -J hf_upload
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@60
#SBATCH --open-mode=append

trap '
    echo "SIGTERM received — requeueing job $SLURM_JOB_ID"
    scontrol requeue $SLURM_JOB_ID
    exit 0
' TERM

python embeddings_upload.py &
PID=$!
wait $PID

echo "Finished at: $(date)"