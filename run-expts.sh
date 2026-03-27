#!/bin/bash

# A script to run experiments, allowing us to keep track of what was done.

#python apply-indices.py ~/projects/farming/salt/images/2024_06_03 ~/projects/farming/salt/results/2024_06_03-normalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n true --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_07_22 ~/projects/farming/salt/results/2024_02_22-normalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n true --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_09_09 ~/projects/farming/salt/results/2024_09_09-normalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n true --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_06_03 ~/projects/farming/salt/results/2024_06_03-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_07_22 ~/projects/farming/salt/results/2024_02_22-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu

python apply-indices.py ~/projects/farming/salt/images/2024_09_09 ~/projects/farming/salt/results/2024_09_09-unnormalized-otsu.csv --indexes ExG ExGR GLI VARI RGBVI DGCI NGBDI BGR GRVI NRI NGI NBI SAVI GMR  --n false --thresholds otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu otsu
