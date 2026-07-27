#!/bin/bash

python3 evaluate_predictions.py --subject S1 --experiment perceived_speech --task wheretheressmoke
python3 evaluate_predictions.py --subject S1_small --experiment perceived_speech --task wheretheressmoke
python3 evaluate_predictions.py --subject S1_noise_s0.25 --experiment perceived_speech --task wheretheressmoke
python3 evaluate_predictions.py --subject S1_noise_s0.5 --experiment perceived_speech --task wheretheressmoke
python3 evaluate_predictions.py --subject S1_noise_s0.7 --experiment perceived_speech --task wheretheressmoke
python3 evaluate_predictions.py --subject S1 --experiment perceived_speech --task wheretheressmoke_noise100_s0.5
python3 evaluate_predictions.py --subject S1_small --experiment perceived_speech --task wheretheressmoke_noise100_s0.5
python3 evaluate_predictions.py --subject S1_noise_s0.25 --experiment perceived_speech --task wheretheressmoke_noise100_s0.5
python3 evaluate_predictions.py --subject S1_noise_s0.5 --experiment perceived_speech --task wheretheressmoke_noise100_s0.5
python3 evaluate_predictions.py --subject S1_noise_s0.7 --experiment perceived_speech --task wheretheressmoke_noise100_s0.5