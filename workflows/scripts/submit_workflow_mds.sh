#!/usr/bin/env bash

WORKFLOW_DIR="/home/exacloud/gscratch/mcweeney_lab/evans/AMLVAE/workflows/MDS"
LOG_DIR="/home/exacloud/gscratch/mcweeney_lab/evans/outputs_/amlvae/MDS/logs"

CONFIGS=(
    configs/config_dim2.yaml
    configs/config_dim8.yaml
    configs/config_dim16.yaml
    configs/config_dim32.yaml
    configs/config_dim64.yaml
    configs/config_dim128.yaml
    configs/config_dim256.yaml
)

mkdir -p "${LOG_DIR}"

for config in "${CONFIGS[@]}"; do
    job_name="$(basename "${config}" .yaml)"
    sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=${job_name}
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=${LOG_DIR}/snakemake_${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/snakemake_${job_name}_%j.err

source ~/.zshrc

cd ${WORKFLOW_DIR}

conda activate amlvae

snakemake --unlock --configfile ./${config}
snakemake -j 1 --rerun-incomplete --configfile ./${config}
EOF
    echo "Submitted ${job_name}"
    sleep 5
done
