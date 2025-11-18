#!/bin/bash
# Try different numbers of steps to find the sweet spot

echo "Testing with 50 steps (10x reduction)..."
python physics_flow_matching/sample_eqm_unconditional.py \
    --checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
    --config configs/darcy_flow_eqm.yaml \
    --num_samples 100 \
    --num_steps 50 \
    --step_size 0.002 \
    --output samples_50steps.npy

echo "Testing with 100 steps (5x reduction)..."
python physics_flow_matching/sample_eqm_unconditional.py \
    --checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
    --config configs/darcy_flow_eqm.yaml \
    --num_samples 100 \
    --num_steps 100 \
    --step_size 0.002 \
    --output samples_100steps.npy

echo "Testing with 200 steps (2.5x reduction)..."
python physics_flow_matching/sample_eqm_unconditional.py \
    --checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
    --config configs/darcy_flow_eqm.yaml \
    --num_samples 100 \
    --num_steps 200 \
    --step_size 0.002 \
    --output samples_200steps.npy
