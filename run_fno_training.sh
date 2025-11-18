#!/bin/bash
# Train FNO with Energy Regularization on Darcy Flow

python train_fno_with_energy.py \
    --data_path data/2D_DarcyFlow_beta1.0_Train.hdf5 \
    --eqm_checkpoint experiments/darcy_flow_eqm/exp_1/saved_state/checkpoint_100.pth \
    --eqm_config configs/darcy_flow_eqm.yaml \
    --train_samples 800 \
    --val_samples 200 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-3 \
    --mse_weight 0.8 \
    --energy_weight 0.2 \
    --energy_temperature 1.0 \
    --checkpoint_save_path checkpoints/fno_with_energy.pth \
    --output_plot fno_training_curves.png \
    --save_every 25 \
    --device cuda
