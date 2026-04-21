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

#python apply-indices.py ~/projects/farming/salt/images/2024_07_22 2024_07_22-normalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n true --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

#python apply-indices.py ~/projects/farming/salt/images/2024_09_09 2024_09_09-normalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n true --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

#python apply-indices.py ~/projects/farming/salt/images/2024_06_03 2024_06_03-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

# Only try unnormalised for the first set of images to begin with
#python apply-indices.py ~/projects/farming/salt/images/2024_07_22 2024_07_22-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

#python apply-indices.py ~/projects/farming/salt/images/2024_09_09 2024_09_09-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

# Threshold median
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n true

#python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n true

#python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n true

#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n false

# Only try unnormalised for the first set of images to begin with
#python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n false

#python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n false

# Threshold 0
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-000.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -n true

# No need to repeat this one since it seems to give all pixels for all
# images since the indexes don't give a value less than zero.

# May want to rerun with strictly > threshold.

# Threshold 10
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-010.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 10 10 10 10 10 10 10 10 10 10 10 10 10 10 -n true

#python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-010.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 10 10 10 10 10 10 10 10 10 10 10 10 10 10 -n true

#python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-010.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 10 10 10 10 10 10 10 10 10 10 10 10 10 10 -n true

#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-theshold-010.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 10 10 10 10 10 10 10 10 10 10 10 10 10 10 -n false

# Only try unnormalised for the first set of images to begin with
#python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n false

#python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t median median median median median median median median median median median median median median -n false

# Threshold 20
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-020.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 20 20 20 20 20 20 20 20 20 20 20 20 20 20 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-020.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 20 20 20 20 20 20 20 20 20 20 20 20 20 20 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-020.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 20 20 20 20 20 20 20 20 20 20 20 20 20 20 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-020.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 20 20 20 20 20 20 20 20 20 20 20 20 20 20 -n false

# Threshold 30
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-030.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 30 30 30 30 30 30 30 30 30 30 30 30 30 30 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-030.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 30 30 30 30 30 30 30 30 30 30 30 30 30 30 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-030.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 30 30 30 30 30 30 30 30 30 30 30 30 30 30 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-030.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 30 30 30 30 30 30 30 30 30 30 30 30 30 30 -n false

# Threshold 40
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-040.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 40 40 40 40 40 40 40 40 40 40 40 40 40 40 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-040.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 40 40 40 40 40 40 40 40 40 40 40 40 40 40 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-040.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 40 40 40 40 40 40 40 40 40 40 40 40 40 40 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-040.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 40 40 40 40 40 40 40 40 40 40 40 40 40 40 -n false

# Threshold 50
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-050.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 50 50 50 50 50 50 50 50 50 50 50 50 50 50 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-050.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 50 50 50 50 50 50 50 50 50 50 50 50 50 50 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-050.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 50 50 50 50 50 50 50 50 50 50 50 50 50 50 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-050.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 50 50 50 50 50 50 50 50 50 50 50 50 50 50 -n false

# Threshold 60
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-060.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 60 60 60 60 60 60 60 60 60 60 60 60 60 60 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-060.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 60 60 60 60 60 60 60 60 60 60 60 60 60 60 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-060.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 60 60 60 60 60 60 60 60 60 60 60 60 60 60 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-060.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 60 60 60 60 60 60 60 60 60 60 60 60 60 60 -n false

# Threshold 70
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-070.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 70 70 70 70 70 70 70 70 70 70 70 70 70 70 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-070.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 70 70 70 70 70 70 70 70 70 70 70 70 70 70 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-070.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 70 70 70 70 70 70 70 70 70 70 70 70 70 70 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-070.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 70 70 70 70 70 70 70 70 70 70 70 70 70 70 -n false

# Threshold 80
# python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-080.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 80 80 80 80 80 80 80 80 80 80 80 80 80 80 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-080.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 80 80 80 80 80 80 80 80 80 80 80 80 80 80 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshld-080.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 80 80 80 80 80 80 80 80 80 80 80 80 80 80 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-080.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 80 80 80 80 80 80 80 80 80 80 80 80 80 80 -n false

# Threshold 90
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-090.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 90 90 90 90 90 90 90 90 90 90 90 90 90 90 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 90 90 90 90 90 90 90 90 90 90 90 90 90 90 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 90 90 90 90 90 90 90 90 90 90 90 90 90 90 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-median.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 90 90 90 90 90 90 90 90 90 90 90 90 90 90 -n false

#Threshold 100
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-100.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 100 100 100 100 100 100 100 100 100 100 100 100 100 100 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-100.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 100 100 100 100 100 100 100 100 100 100 100 100 100 100 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-100.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 100 100 100 100 100 100 100 100 100 100 100 100 100 100 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-100.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 100 100 100 100 100 100 100 100 100 100 100 100 100 100 -n false

# Threshold 110
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-110.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 110 110 110 110 110 110 110 110 110 110 110 110 110 110 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-110.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 110 110 110 110 110 110 110 110 110 110 110 110 110 110 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-110.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 110 110 110 110 110 110 110 110 110 110 110 110 110 110 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-110.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 110 110 110 110 110 110 110 110 110 110 110 110 110 110 -n false

# Threshold 120
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-120.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 120 120 120 120 120 120 120 120 120 120 120 120 120 120 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-120.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 120 120 120 120 120 120 120 120 120 120 120 120 120 120 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-120.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 120 120 120 120 120 120 120 120 120 120 120 120 120 120 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-120.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 120 120 120 120 120 120 120 120 120 120 120 120 120 120 -n false

# Threshold 130
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-130.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 130 130 130 130 130 130 130 130 130 130 130 130 130 130 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-130.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 130 130 130 130 130 130 130 130 130 130 130 130 130 130 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-130.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 130 130 130 130 130 130 130 130 130 130 130 130 130 130 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-130.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 130 130 130 130 130 130 130 130 130 130 130 130 130 130 -n false

# Threshold 140
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-140.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 140 140 140 140 140 140 140 140 140 140 140 140 140 140 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-140.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 140 140 140 140 140 140 140 140 140 140 140 140 140 140 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-140.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 140 140 140 140 140 140 140 140 140 140 140 140 140 140 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-140.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 140 140 140 140 140 140 140 140 140 140 140 140 140 140 -n false

# Threshold 150
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-150.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 150 150 150 150 150 150 150 150 150 150 150 150 150 150 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-150.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 150 150 150 150 150 150 150 150 150 150 150 150 150 150 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-150.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 150 150 150 150 150 150 150 150 150 150 150 150 150 150 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-150.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 150 150 150 150 150 150 150 150 150 150 150 150 150 150 -n false

# Threshold 160
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-160.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 160 160 160 160 160 160 160 160 160 160 160 160 160 160 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-160.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 160 160 160 160 160 160 160 160 160 160 160 160 160 160 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-160.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 160 160 160 160 160 160 160 160 160 160 160 160 160 160 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-160.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 160 160 160 160 160 160 160 160 160 160 160 160 160 160 -n false

# Threshold 170
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-170.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 170 170 170 170 170 170 170 170 170 170 170 170 170 170 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-170.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 170 170 170 170 170 170 170 170 170 170 170 170 170 170 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-170.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 170 170 170 170 170 170 170 170 170 170 170 170 170 170 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-170.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 170 170 170 170 170 170 170 170 170 170 170 170 170 170 -n false

# Threshold 180
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-180.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 180 180 180 180 180 180 180 180 180 180 180 180 180 180 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-180.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 180 180 180 180 180 180 180 180 180 180 180 180 180 180 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-180.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 180 180 180 180 180 180 180 180 180 180 180 180 180 180 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-180.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 180 180 180 180 180 180 180 180 180 180 180 180 180 180 -n false

# Threshold 190
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-190.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 190 190 190 190 190 190 190 190 190 190 190 190 190 190 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-190.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 190 190 190 190 190 190 190 190 190 190 190 190 190 190 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-190.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 190 190 190 190 190 190 190 190 190 190 190 190 190 190 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-190.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 190 190 190 190 190 190 190 190 190 190 190 190 190 190 -n false

# Threshold 200
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-200.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 200 200 200 200 200 200 200 200 200 200 200 200 200 200 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-200.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 200 200 200 200 200 200 200 200 200 200 200 200 200 200 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-200.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 200 200 200 200 200 200 200 200 200 200 200 200 200 200 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-200.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 200 200 200 200 200 200 200 200 200 200 200 200 200 200 -n false

# Threshold 210
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-210.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 210 210 210 210 210 210 210 210 210 210 210 210 210 210 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-210.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 210 210 210 210 210 210 210 210 210 210 210 210 210 210 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-210.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 210 210 210 210 210 210 210 210 210 210 210 210 210 210 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-210.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 210 210 210 210 210 210 210 210 210 210 210 210 210 210 -n false

# Threshold 220
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-220.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 220 220 220 220 220 220 220 220 220 220 220 220 220 220 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-220.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 220 220 220 220 220 220 220 220 220 220 220 220 220 220 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-220.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 220 220 220 220 220 220 220 220 220 220 220 220 220 220 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-220.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 220 220 220 220 220 220 220 220 220 220 220 220 220 220 -n false

# Threshold 230
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-230.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 230 230 230 230 230 230 230 230 230 230 230 230 230 230 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-230.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 230 230 230 230 230 230 230 230 230 230 230 230 230 230 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-230.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 230 230 230 230 230 230 230 230 230 230 230 230 230 230 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-230.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 230 230 230 230 230 230 230 230 230 230 230 230 230 230 -n false

# Threshold 240
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-240.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 240 240 240 240 240 240 240 240 240 240 240 240 240 240 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-240.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 240 240 240 240 240 240 240 240 240 240 240 240 240 240 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-240.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 240 240 240 240 240 240 240 240 240 240 240 240 240 240 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-240.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 240 240 240 240 240 240 240 240 240 240 240 240 240 240 -n false

# Threshold 250
#python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-normalised-threshold-250.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 250 250 250 250 250 250 250 250 250 250 250 250 250 250 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_07_22/ 2024_07_22-normalised-threshold-250.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 250 250 250 250 250 250 250 250 250 250 250 250 250 250 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_09_09/ 2024_09_09-normalised-threshold-250.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 250 250 250 250 250 250 250 250 250 250 250 250 250 250 -n true

python apply-indices.py ~/projects/farming/salt/images/2024_06_03/ 2024_06_03-unnormalised-threshold-250.csv -i ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR -t 250 250 250 250 250 250 250 250 250 250 250 250 250 250 -n false
