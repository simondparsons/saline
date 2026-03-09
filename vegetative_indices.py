# vegetative_indices.py
#
# Code to compute various vegetative indices
#
# Simon Parsons
# University of Lincoln
# 26-03-06

# Vegetative indices as a library. Largely my own work, though Otsu
# came frrom Claude Sonnet 4.5 (which basicaly copied it from the
# OpenCV web page).

# Necesary libraries
import argparse
import numpy as np
import cv2 as cv

# Computing vegetative indices. These functions work at the pixel level.

# Definitions for ExG, ExGR, GLI and VARI come from: L. Rosen,
# P. M. Ewing, and B. C. Runk, RGB-based indices for estimating cover
# crop biomass, nitrogen content, and carbon:nitrogen ratio, Agronomy
# Journal, 116(6):3070-3080, 2024.

# Excess green (ExG)
def computeExG(b, g, r):
    # Convert to float64 first to avoid overflow
    b = b.astype(np.float64)
    g = g.astype(np.float64)
    r = r.astype(np.float64)
    
    return (((2 * g) - b) - r)

# Excess green minus excess red (ExGR)
def computeExGR(b, g, r):
    # Convert to float64 first to avoid overflow
    b = b.astype(np.float64)
    g = g.astype(np.float64)
    r = r.astype(np.float64)
    
    return (((3 * g) - (2.4 * r)) - b)

# Green leaf index (GLI)
def computeGLI(b, g, r):
    # Convert to float64 first to avoid overflow
    b = b.astype(np.float64)
    g = g.astype(np.float64)
    r = r.astype(np.float64)
    
    # Avoid division by zero
    denominator = ((2 * g) + b) + r
    if denominator == 0:
        denominator = 1e-10
    return ((((2 * g) - b) - r) / denominator)

#  Visible atmospherically resistant index (VARI)
def computeVARI(b, g, r):
    # Convert to float64 first to avoid overflow
    b = b.astype(np.float64)
    g = g.astype(np.float64)
    r = r.astype(np.float64)
    
    denominator = (g + r) - b
    if denominator == 0:
        denominator = 1e-10
    return ((g - r) / denominator)

# Normalize across B, G and R bands. In theory this removes effects
# due to illumination.
def normalizeBands(img):

    b = img[:,:,0].astype(np.float64) # get blue channel
    g = img[:,:,1].astype(np.float64) # get green channel
    r = img[:,:,2].astype(np.float64) # get red channel
    
    # Calculate sum with scaling to avoid overflow
    sum_channels = b + g + r
    
    # Avoid division by zero
    sum_channels = np.where(sum_channels == 0, 1e-10, sum_channels)
    
    # Normalize each channel
    b_n =  (b / sum_channels)
    g_n =  (g / sum_channels)
    r_n =  (r / sum_channels)

    # Note, need to convert these back to uint8 when used.
    return b_n, g_n, r_n

# The same, but returns a normalized image
def normalizeImage(img):

    b = img[:,:,0].astype(np.float64) # get blue channel
    g = img[:,:,1].astype(np.float64) # get green channel
    r = img[:,:,2].astype(np.float64) # get red channel
    
    # Calculate sum with scaling to avoid overflow
    sum_channels = b + g + r
    
    # Avoid division by zero
    sum_channels = np.where(sum_channels == 0, 1e-10, sum_channels)
    
    # Normalize each channel
    b_n = (b / sum_channels)
    g_n = (g / sum_channels)
    r_n = (r / sum_channels)

    # Stack channels back together
    normalized = np.stack([b_n, g_n, r_n], axis=2)
    
    # Scale to 0-255 range and convert to uint8
    normalized_scaled = cv.normalize(normalized, None, 0, 255, cv.NORM_MINMAX)
    normalized_uint8 = normalized_scaled.astype(np.uint8)
    
    return normalized_uint8

# Calculate an index across an image. Takes the relevant pixel-level
# function as input.
#
# Can't currently get this to be called properly from outside the module.
def computeIndex(img, indexFunc):
    b = img[:,:,0] # get blue channel
    g = img[:,:,1] # get green channel
    r = img[:,:,2] # get red channel

    newImg = np.zeros(b.shape)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newImg[i][j] = indexFunc(b[i][j], g[i][j], r[i][j])

    # Return images that look like standard images
    imgScaled = cv.normalize(newImg, None, 0, 255, cv.NORM_MINMAX)
    imgUint8 = imgScaled.astype(np.uint8)
    return imgUint8

# Instead we have some specialised functions that call it.
def computeExGImage(img):
    return computeIndex(img, computeExG)

def computeExGRImage(img):
    return computeIndex(img, computeExGR)

def computeGLIImage(img):
    return computeIndex(img, computeGLI)

def computeVARIImage(img):
    return computeIndex(img, computeVARI)

# Apply a threshold to a single channel image
def applyThreshold(img, thresh):
    newImg = np.zeros(img.shape)
    pixelCount = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # If value is above threshold
            if img[i][j] >= thresh:
                newImg[i][j] = 255
                pixelCount = pixelCount + 1
                
    return newImg, pixelCount

# Compute the Otsu theshold for an image. Needs a standard OpenCV
# image (i.e. uint8)
def calculateOtsuThreshold(img):

    # Convert to grayscale if image is color
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply Otsu's thresholding, returns a float
    otsu_threshold, _ = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    return otsu_threshold

