#!/bin/bash

export HUGGING_FACE_HUB_TOKEN=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

module load python

mamba activate /n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/envs/diffusers

python generate_an_image.py