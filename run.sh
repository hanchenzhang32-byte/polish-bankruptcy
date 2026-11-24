#!/bin/sh
# Needs the five ARFF files from the UCI zip in data/raw/ (see README).
set -e
mkdir -p results
python3 scripts/load_data.py data/raw data/companies.csv > results/load_data_output.txt
python3 scripts/train.py data/companies.csv results > results/train_output.txt
echo "Done. See results/"
