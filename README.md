# saline
Code to analyse results of some salinity tests. Image processing on RGB images.

## Key files

apply-indices.py
Code to call indices on a directory of files. Handles the loading of files and structuring of output.

vegetative-indices.py
Implements a number (currently 12) of vegetative indices, both at the pixel level, and across images, inclduing allowing thresholds to be applied based on the indices returning thresholded images and pixel counts.

### Notes

Indices need to be fuly tested, 2G - B - R index needs to be implemented, and apply-indices.py should be modified to use the dictionary-based dispatcher.

## Other files

threshold.py
Applies a couple of simple thresholds and outputs counts of pixels that meet the thresholds.

gpu-hooks.py
Provides a way to invoke gpu computations on machines with CUDA.
