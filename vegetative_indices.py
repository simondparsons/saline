# vegetative_indices.py
#
# Code to compute various vegetative indices
#
# Simon Parsons
# University of Lincoln
# 26-03-06

# Vegetative indices as a library. This started with my own work, with
# Otsu from Claude Sonnet 4.5 (which basically copied it from the
# OpenCV web page), but then Chat GPT wrote a bunch more of the
# functions for me.

# Necesary libraries
import argparse
import numpy as np
import cv2 as cv

# Use CuPy for GPU support if GPU is available
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Computing vegetative indices. These functions work at the pixel level.

# Definitions for ExG, ExGR, GLI and VARI come from: L. Rosen,
# P. M. Ewing, and B. C. Runk, RGB-based indices for estimating cover
# crop biomass, nitrogen content, and carbon:nitrogen ratio, Agronomy
# Journal, 116(6):3070-3080, 2024.
#
# These indices were used with thresholds.

# Epsilon to use to avoid division by zero.
EPS = 1e-10

# =========================
# Excess green (ExG)
# =========================
def computeExG(b, g, r):

    return (((2 * g) - b) - r)

# =========================
# Excess green minus excess red (ExGR)
# =========================
def computeExGR(b, g, r):
    
    return (((3 * g) - (2.4 * r)) - b)

# =========================
# Green leaf index (GLI)
# =========================
def computeGLI(b, g, r):
    # Avoid division by zero
    denominator = ((2 * g) + b) + r + EPS
    
    return ((((2 * g) - b) - r) / denominator)

# =========================
#  Visible atmospherically resistant index (VARI)
# =========================
def computeVARI(b, g, r):    
    denominator = (g + r) - b + EPS

    return ((g - r) / denominator)

# Checked against: A. Rossi, S. Tavarini, M. Tognoni, L. G. Angelini,
# C. Clemente, L. Caturegli, Reliable NDVI estimation in wheat using
# low-Cost UAV-derived RGB vegetation indices, Smart Agricultural
# Technology, 12:101452, 2025.
#
# These indices were used to compute an average value across parts of an image

# =========================
# Red Green Blue Vegetation Index (RGBVI)
# =========================
def computeRGBVI(b, g, r):

    return (g**2 - (r * b)) / (g**2 + (r * b) + EPS)

# =========================
# Dark Green Colour Index (DGCI)
# =========================
def computeDGCI(b, g, r):
    # DGCI is defined in Rossi et al. in terms of HSV
    h, s, v = rgb_to_hsv(r, g, b)
    
    #h = np.float64(h) 
    #s = np.float64(s)
    # Don't need v in the calculation
    return (((h - 60)/60) + (1 - s) + (1 - b)) / 3

# =========================
# Normalized Green Blue Difference Index (NGBDI)
# =========================
def computeNGBDI(b, g, r):

    return (g - b) / (g + b + EPS)

# =========================
# 2G - B - R Index (BGR)
# literally just 2*g - b - r
# =========================
def computeBGR(b, g, r):

    return (2 * g) - b - r

# =========================
# Green Red Vegetation Index (GRVI)
# =========================
def computeGRVI(b, g, r):

    return (g - r) / (g + r + EPS)

# =========================
# Normalized Redness Intensity (NRI)
# =========================
def computeNRI(b, g, r):

    return r / (r + g + b + EPS)

# =========================
# Normalized Greenness Intensity (NGI)
# =========================
def computeNGI(b, g, r):

    return g / (r + g + b + EPS)

# =========================
# Normalized Blueness Intensity (NBI)
# =========================
def computeNBI(b, g, r):

    return b / (r + g + b + EPS)

# =========================
# Soil Adjusted Vegetation Index (SAVI – RGB-based)
# =========================
#
# This matches the definition of SAVI in Rossi et al. but the original
# definition from Huete, A.R., 1988. A soil-adjusted vegetation index
# (SAVI). Remote sensing of environment, 25(3), pp.295-309. has NIR in
# place of G.
#
def computeSAVI(b, g, r, L=0.5):

    return ((g - r) / (g + r + L + EPS)) * (1 + L)

# =========================
# Green Minus Red (GMR)
# =========================
def computeGMR(b, g, r):

    return g - r

# =========================
# Normalization
# =========================

# Normalize across B, G and R bands. In theory this removes effects
# due to illumination.
def normalizeBands(img):

    b = img[:,:,0].astype(np.float64) # get blue channel
    g = img[:,:,1].astype(np.float64) # get green channel
    r = img[:,:,2].astype(np.float64) # get red channel
    sum_channels = b + g + r
    
    # Avoid division by zero
    sum_channels = np.where(sum_channels == 0, 1e-10, sum_channels)
    
    # Normalize each channel
    b_n =  (b / sum_channels)
    g_n =  (g / sum_channels)
    r_n =  (r / sum_channels)

    # Note, need to convert these back to uint8 when used, as below.
    return b_n, g_n, r_n

# The same, but returns a normalized image. This is what is called
# from apply-indices. Converts back to uint8 so that the normalized
# image is the same format as before normalization.
def normalizeImage(img):

    b = img[:,:,0].astype(np.float64) # get blue channel
    g = img[:,:,1].astype(np.float64) # get green channel
    r = img[:,:,2].astype(np.float64) # get red channel
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

# =========================
# Calculate an index across an image. 
# =========================

# This is what is invoked by the dispatcher below. Uses GPU/CuPy if
# possible, else falls back to the legacy version (see below).
#

# GPU version
#
# The wrinkle with this is the need to explicitly convert to numpy
# arrays where we need to use those in indexFunc and downstream.
def computeIndexGPU(img, indexFunc):
    if not GPU_AVAILABLE:
        return computeIndex(img, indexFunc)

    # Use CuPy to get the benefit of GPU
    img_gpu = cp.asarray(img)
    b = img_gpu[:, :, 0]
    g = img_gpu[:, :, 1]
    r = img_gpu[:, :, 2]
    result_gpu = indexFunc(b, g, r)

    # Scale and turn into unit8 so that the results look like a normal
    # image(for example for use with Otsu thresholding)
    newImg = cp.asnumpy(result_gpu)
    imgScaled = cv.normalize(newImg, None, 0, 255, cv.NORM_MINMAX)
    imgUint8 = imgScaled.astype(np.uint8)
    return imgUint8    

# CPU-only versions
#
# Original, pixel, by pixel. Works but slow
def computeIndexOld(img, indexFunc):
    b = img[:,:,0].astype(np.float64) # get blue channel
    g = img[:,:,1].astype(np.float64) # get green channel
    r = img[:,:,2].astype(np.float64) # get red channel

    newImg = np.zeros(b.shape).astype(np.float64)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newImg[i][j] = indexFunc(b[i][j], g[i][j], r[i][j])

    # Return images that look like standard images
    imgScaled = cv.normalize(newImg, None, 0, 255, cv.NORM_MINMAX)
    imgUint8 = imgScaled.astype(np.uint8)
    return imgUint8

# Using numpy vectorization 
def computeIndex(img, indexFunc):
    # Convert to float64 so that we don't need to do it in indexFunc
    b = img[:,:,0].astype(np.float64) # get blue channel
    g = img[:,:,1].astype(np.float64) # get green channel
    r = img[:,:,2].astype(np.float64) # get red channel

    # Vectorize the index function
    vectorizedFunc = np.vectorize(indexFunc)

    # Apply to entire array
    newImg = vectorizedFunc(b, g, r)
    
    # Return images that look like standard images
    imgScaled = cv.normalize(newImg, None, 0, 255, cv.NORM_MINMAX)
    imgUint8 = imgScaled.astype(np.uint8)
    return imgUint8

# =========================
# Dispatcher
# =========================

# Allows one function to be called from outside the package, passing
# the relevant pixel-wise function to computeIndex, eliminating the
# need for one function per index to do this.
#
# Note that there should be a way of combining this with the similar
# list in apply-indices.py so that we only need to name each index
# once.

INDEX_FUNCTIONS = {
    "ExG": computeExG,
    "ExGR": computeExGR,
    "GLI":  computeGLI,
    "VARI": computeVARI,
    "RGBVI": computeRGBVI,
    "GLI": computeGLI,
    "DGCI": computeDGCI,
    "NGBDI": computeNGBDI,
    "BGR": computeBGR,
    "GRVI": computeGRVI,
    "NRI": computeNRI,
    "NGI": computeNGI,
    "NBI": computeNBI,
    "SAVI": computeSAVI,
    "GMR": computeGMR,
}

def computeIndexByName(img, index_name):
    """Compute a vegetation index by name.

    Parameters
    ----------
    img : ndarray
        Image array (H, W, 3), assumed BGR or RGB consistently
    index_name : str
        Key from INDEX_FUNCTIONS

    Returns
    -------
    ndarray
        Computed index image

    Given the nature of the computation, a GPU should speed things up
    a lot, so we include code to use a GPU is one is available.
    """
    
    if index_name not in INDEX_FUNCTIONS:
        raise ValueError(f"Unknown index '{index_name}'. "
                         f"Available indices: {list(INDEX_FUNCTIONS.keys())}")

    #return computeIndex(img, INDEX_FUNCTIONS[index_name])
    return computeIndexGPU(img, INDEX_FUNCTIONS[index_name])

def computeMultipleIndices(img, index_names):
    """
    Compute multiple vegetation indices in one call.

    Returns
    -------
    dict[str, ndarray]
        Dictionary mapping index name to index image
    """
    return {name: computeIndexByName(img, name) for name in index_names}

# =========================
# Thresholding
# =========================

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

# =========================
# Averaging etc
# =========================

# For some indices we want the average over the image, As for
# applyThreshold, this expects to be called on the result of computing
# the index, so we have a "grayscale image" as input, where each pixel
# is the index value (though it is float64 not a uint8)
#
# Similarly we may want max and min values (though for now it doesn't
# seem this is helpful, min is 0 and max is 255)

def summaryValues(img):
    return np.mean(img), np.median(img), np.max(img), np.min(img), 

# =========================
# RGB to HSV using OpenCV
# =========================

def rgb_to_hsv(r, g, b):
    # OpenCV expects BGR and values in range [0,255]
    if GPU_AVAILABLE:
        print("Convert to uint8")
        r = cp.asnumpy(r).astype(np.uint8) 
        g = cp.asnumpy(b).astype(np.uint8)
        b = cp.asnumpy(g).astype(np.uint8)
        bgr_pixel = np.uint8([[[b, g, r]]])
        print("Calling open CV")
        hsv_pixel = cv.cvtColor(bgr_pixel, cv.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]
        print("Convert back to float64")
        h = cp.float64(h.get()) 
        s = cp.float64(s.get())
        v = cp.float64(v.get())
    else:
        bgr_pixel = np.uint8([[[b, g, r]]])
        hsv_pixel = cv.cvtColor(bgr_pixel, cv.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]
        # Now convert back into float64 so we don't need to do that in
        # the index function.
        h = np.float64(h) 
        s = np.float64(s)
        v = np.float64(s)

    return h, s, v
