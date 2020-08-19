#!/bin/bash
#SBATCH --job-name=python
#SBATCH --output=sbatch.out
#SBATCH --error=sbatch.err
#SBATCH --time=30:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1

module load Anaconda3

python test.py
