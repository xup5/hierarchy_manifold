#!/bin/bash
#SBATCH --job-name=animal_dataset
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00
#SBATCH --mem=128G
#SBATCH --output=/n/home13/xupan/out/%x_%j.out
#SBATCH --error=/n/home13/xupan/err/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=xupan@fas.harvard.edu

echo "started"

module load python

mamba activate /n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/envs/diffusers

cd /n/home13/xupan/Projects/hierarchy_manifold/hierarchy_manifold/stable_diffusion

python generate_animal_order_species.py 