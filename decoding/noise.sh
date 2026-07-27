#!/bin/bash

python3 add_train_voxel_noise.py --subject S1 --noise_scale 0.5
python3 add_train_voxel_noise.py --subject S1 --noise_scale 0.7
python3 add_train_voxel_noise.py --subject S2 --noise_scale 0.25
python3 add_train_voxel_noise.py --subject S2 --noise_scale 0.5
python3 add_train_voxel_noise.py --subject S2 --noise_scale 0.7
python3 add_train_voxel_noise.py --subject S3 --noise_scale 0.25
python3 add_train_voxel_noise.py --subject S3 --noise_scale 0.5
python3 add_train_voxel_noise.py --subject S3 --noise_scale 0.7
