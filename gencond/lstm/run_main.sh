#!/bin/bash
#SBATCH --partition=cui-gpu-EL7
#SBATCH -J lstm_guacamol
#SBATCH --mem=12000
#SBATCH -o jobname-out.o%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 95:00:00
#BATCH --mail-user=amina.mollaysa@gmail.com
module load GCC/6.3.0-2.27 Singularity/2.4.2
#module load CUDA

srun singularity exec --nv /home/aminanm0/jao_atalaya  python train_zinc.py 
