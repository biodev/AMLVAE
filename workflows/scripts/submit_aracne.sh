#!/usr/bin/env bash

REPO_ROOT="/home/exacloud/gscratch/mcweeney_lab/evans/AMLVAE"
SCRIPT="${REPO_ROOT}/workflows/scripts/aracne.py"
LOG_DIR="/home/exacloud/gscratch/mcweeney_lab/evans/outputs_/amlvae/MDS/logs"

mkdir -p "${LOG_DIR}"

sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=aracne
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=${LOG_DIR}/aracne_%j.out
#SBATCH --error=${LOG_DIR}/aracne_%j.err

source ~/.zshrc

conda activate viper

python ${SCRIPT} \\
    --n-bootstraps 100 \\
    --threads 32 \\
    --xmx 50G \\
    --top-k 20000 \\
    --force-rerun
EOF

echo "Submitted aracne job"
