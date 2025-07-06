#! /usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# need to modify purified samples dataroot to your own

# ModelNet40
python make_data/make_mix_attack.py
python make_data/make_single_attack.py
python make_data/adaptive/make_mix_attack.py
python make_data/adaptive/make_single_attack.py

# ScanobjectNN
python scanobjectnn/attack_recons/make_mix_attack.py
python scanobjectnn/attack_recons/make_single_attack.py
python scanobjectnn/attack_recons/adaptive/make_mix_attack.py
python scanobjectnn/attack_recons/adaptive/make_single_attack.py