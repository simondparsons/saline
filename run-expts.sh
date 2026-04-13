#!/bin/bash

# A script to run experiments for the saline soils data, allowing us
# to keep track of what was done.
#
# Simon Parsons
# University of Lincoln
# 26-04-11
#
# Each run/experiment involves running the full set of vegetative
# indices with a specific threshold on one of the "prior to harvest"
# set of images. There are three of tehse, and we set up to run with
# both normalizaed and un-normalized images (though at the time of
# writing we have yet to see if un-normalized images are nay good).
#
# The number output for each index is the number of pixels >= the
# threshold.
#
# The different thresholds are denoted in comments.
#
# Where experiments have been run, the relevant commands have been
# commented out.

# Threshold otsu
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03 ~/projects/farming/salt/results/2024_06_03-normalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n true --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_07_22 2024_07_22-normalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n true --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_09_09 2024_09_09-normalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n true --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_06_03 2024_06_03-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_07_22 2024_07_22-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_09_09 2024_09_09-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

# Threshold median
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n false

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n false

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n false

# Threshold 0
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-000.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -n true

# No need to repeat this one since it seems to give all pixels for all
# images since the indexes don't give a value less than zero.

# May want to rerun with strictly > threshold.

# Threshold 10
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-010.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 10 10 10 10 10 10 10 10 10 10 10 10 10 10 -n true

# Threshold 20
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-020.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 20 20 20 20 20 20 20 20 20 20 20 20 20 20 -n true

# Threshold 30
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-030.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 30 30 30 30 30 30 30 30 30 30 30 30 30 30 -n true

# Threshold 40
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-040.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 40 40 40 40 40 40 40 40 40 40 40 40 40 40 -n true

# Threshold 50
python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-050.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 50 50 50 50 50 50 50 50 50 50 50 50 50 50 -n true
