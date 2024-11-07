#!/bin/bash

module load python

mamba activate /n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/envs/diffusers

cd /n/home13/xupan/Projects/hierarchy_manifold/hierarchy_manifold/stable_diffusion

python generate_an_image.py