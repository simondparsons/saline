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
#
#
# DGCI is defined in Rossi et al. in terms of HSV. We don't need
# the v value
#
# In order to calculate this index on a GPU, we have two versions.

# Because of the call to rgb_to_hsv(), this version can't be run on a
# GPU, and it means that computingthis index is much slower than the
# other on a GPU-equipped machine: ~30 seconds compared to under a
# second per image.
def computeDGCI(b, g, r):
    h, s, _ = rgb_to_hsv(r, g, b)
    
    return (((h - 60)/60) + (1 - s) + (1 - b)) / 3

# So, we have another version specifically for running on a GPU. This
# version accepts just the three parameters it needs and expects the
# RGB to HSV conversion to be done elsewhere. (Using the mixed set of
# parameters is a bit of hack to keep the dispatcher code the same.
def computeDGCI_GPU(b, h, s):

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
    else:
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

# Using numpy vectorization. What we do when we don't have a GPU
# available.
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
    """
    # Given the nature of the computation, a GPU should speed things
    # up a lot, so we include code to use a GPU if one is
    # available. 
    if index_name not in INDEX_FUNCTIONS:
        raise ValueError(f"Unknown index '{index_name}'. "
                         f"Available indices: {list(INDEX_FUNCTIONS.keys())}")
    elif index_name == "DGCI":
    # For DGCI we need a mix of RGB and HSV values, so we compute the
    # HSV values first, in a GPU-compatible way, and then dispatch in
    # the normal way, just sending the necessary parameters.
        if GPU_AVAILABLE:
            hs_img = bgr_to_hsv(img)
            # Now need to build an "image" with the relevant channels for DGCI
            dgci_img = constructDGCIImage(img, hs_img)
            return computeIndexGPU(dgci_img, computeDGCI_GPU)
        else:
            return computeIndex(img, INDEX_FUNCTIONS[index_name])
    else:
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

# Apply a threshold to a single channel image.
def applyThreshold(img, thresh):
    # Produce a binary mask from the image
    newImg = np.where(img >= thresh, 255, 0)
    pixelCount = (img >= thresh).sum()
    
    return newImg, pixelCount

# Original, pixel by pixel version
def applyThresholdOld(img, thresh):
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
# This does not currently allow for GPU usage, preventing its use in
# calculating DGCI on a GPU. Only called when running on a CPU.
def rgb_to_hsv(r, g, b):
    # OpenCV expects BGR and values in range [0,255]
    bgr_pixel = np.uint8([[[b, g, r]]])
    hsv_pixel = cv.cvtColor(bgr_pixel, cv.COLOR_BGR2HSV)
    h, s, v = hsv_pixel[0][0]
    # Now convert back into float64 so we don't need to do that in
    # the index function.
    h = np.float64(h) 
    s = np.float64(s)
    v = np.float64(v)
    # Finally convert back to proper HSV values (since we assume these
    # are what are used in DGCI) rather than the odd values used by
    # OpenCV (see https://docs.opencv.org/3.4/d8/d01/
    #                         group__imgproc__color__conversions.html)
    # h is doubled, and s and v scaled to be between 0 and 100
    h = np.float64(h * 2) 
    s = np.float64(s / 2.55)
    v = np.float64(v / 2.55)
    return h, s, v

# =========================
# RGB to HSV directly
# =========================
# Since we can't run the OpenCV code on a GPU, we need to calculate
# directly using CuPy. The function name reflects the fact that since
# we read images in using OpenCV, we are always dealing with BGR
# rather than RGB.
#
# Expects a regular image format and returns a CuPy array a CuPy array
# of shape (H, W, 3), float32.
#             - H: Hue        (0–360)
#             - S: Saturation (0–100)
#             - V: Value      (0–100)
# since these are the standard HSV values rather than the ones
# returned by OpenCV where H is in (0-179), so that it fits in 0-255,
# and S and V are both in (0-255).
def bgr_to_hsv(img):
    image = cp.asarray(img)
    image = image.astype(cp.float32)

    # Start by normalizing the B, G, R values to 0–1
    #if image.max() > 1.0:
    image = image / 255.0

    # Unpack BGR channels
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    cmax = cp.maximum(cp.maximum(r, g), b)  # Value
    cmin = cp.minimum(cp.minimum(r, g), b)
    delta = cmax - cmin

    # --- Value ---
    v = cmax * 100

    # --- Saturation ---
    #s = cp.zeros_like(cmax)

    s = cp.where(cmax != 0, (delta / cmax) * 100, cp.zeros_like(cmax))

    # --- Hue ---
    h = cp.zeros_like(cmax)

    # Hue when cmax == r, here h is zero
    mask_r = (cmax == r) #& (delta != 0)
    h = cp.where(mask_r, ((g - b) / (delta + EPS)) % 6, h)

    # Hue when cmax == g, here h is set by the previous cp.where
    mask_g = (cmax == g) #& (delta != 0)
    h = cp.where(mask_g, ((b - r) / (delta + EPS)) + 2, h)

    # Hue when cmax == b, here h is set by the previous cp.wheres
    mask_b = (cmax == b) #& (delta != 0)
    h = cp.where(mask_b, ((r - g) / (delta + EPS)) + 4, h)

    h = h * 60 

    #Add 360 where h is less than 0
    mask_h = (h < 0)
    h = cp.where(mask_h, h + 360, h)
    #if h < 0:
    #    h = h + 360
        
    # Stack into (H, W, 3) HSV image
    hsv = cp.stack([h, s, v], axis=-1)

    return hsv

# DCGI expects an "image" which is made up of a mix of RGB and HSV,
# namely (b, h, s).
#
# img is a regular RGB format, hs_img is a CuPY array (from
# bgr_to_hsv)
def constructDGCIImage(img, hs_img):
    # Create a CuPy version of the BGR image
    cp_img = cp.asarray(img)
    cp_img = cp_img.astype(cp.float32)
    # Extract relevant channels
    b = cp_img[:, :, 0]
    h = hs_img[:, :, 0]
    s = hs_img[:, :, 1]

    bhs = cp.stack([b, h, s], axis=-1)

    return bhs
