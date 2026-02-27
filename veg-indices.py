# Veg-indices.py
#
# Code to compute various vegetative indices
#
# Simon Parsons
# University of Lincoln
# 27-02-23

# Necesary libraries
import argparse
import numpy as np
#import matplotlib.pyplot as plt
import cv2 as cv


# Normalize across B, G and R bands. In theory this removes effects
# due to illumination.
def normalizeBands(img):
    b = img[:,:,0] # get blue channel
    g = img[:,:,1] # get green channel
    r = img[:,:,2] # get red channel

    # Empty arrays for the normalized channels
    b_n = np.zeros(b.shape)
    g_n = np.zeros(g.shape)
    r_n = np.zeros(r.shape)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Mess with the sum like this to avoid overflow errors
            sum = (0.25*b[i][j] + 0.25*g[i][j] + 0.25*r[i][j])

            if sum != 0:
                b_n[i][j] = 0.25*(b[i][j]/sum)
                g_n[i][j] = 0.25*(g[i][j]/sum)
                r_n[i][j] = 0.25*(r[i][j]/sum)

    return b_n, g_n, r_n

# Computing vegetative indices

# Definitions for ExG, ExGR, GLI and VARI come from: L. Rosen,
# P. M. Ewing, and B. C> Runk, RGB-based indices for estimating cover
# crop biomass, nitrogen content, and carbon:nitrogen ratio, Agronomy
# Journal, 116(6):3070-3080, 2024.

# Excess green (ExG)
def computeExG(b, g, r):
    return (((2 * g) - b) - r)

# Excess green minus excess red (ExGR)
def computeExGR(b, g, r):
    return (((3 * g) - 2.4 * r) - b)

# Green leaf index (GLI)
def computeGLI(b, g, r):
    return ((((2 * g) - b) - r) / (((2 * g) + b) + r))

#  Visible atmospherically resistant index (VARI)
def computeVARI(b, g, r):
    return ((g - r) / ((g + r) - b))

# Calculate an index across an image. Takes the relevant pixel-level
# function as input.

def computeIndex(img, indexFunc):
    b = img[:,:,0] # get blue channel
    g = img[:,:,1] # get green channel
    r = img[:,:,2] # get red channel

    newImg = np.zeros(b.shape)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            newImg[i][j] = indexFunc(b[i][j], g[i][j], r[i][j])

    return newImg

# Apply a threshold to a single channel image
def applyThreshold(img, thresh):
    newImg = np.zeros(img.shape)
    pixelCount = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # If values is above threshold
            if img[i][j] >= thresh:
                newImg[i][j] = 255
                pixelCount = pixelCount + 1
                
    return newImg, pixelCount

def main():
    # Start with our standard command line stuff
    #
    # Allow command line specification of the input file
    parser = argparse.ArgumentParser(description='Simple threshold application for RGB images.')
    parser.add_argument('--input', help='Path to input image.')
    parser.add_argument('--display', help='If we display images.')
    args = parser.parse_args()

    # Load the image file identified in the command line
    src = cv.imread(args.input)
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)

    # Do we display images?
    if args.display != None:
        display = True
    else:
        display = False

    # Show initial image
    if display:
        cv.namedWindow("Original image", cv.WINDOW_NORMAL)
        cv.imshow("Original image", src)

    # Split in to B, G and R channels. Could use bgr_planes but the internet says this is more efficient.
    b = src[:,:,0] # get blue channel
    g = src[:,:,1] # get green channel
    r = src[:,:,2] # get red channel

    # First step is to normalize bands.
    b_normal, g_normal ,r_normal  = normalizeBands(src)

    if display:
        cv.namedWindow("B Channel", cv.WINDOW_NORMAL)
        cv.imshow("B Channel", b_normal)
        cv.namedWindow("G Channel", cv.WINDOW_NORMAL)
        cv.imshow("G Channel", g_normal)
        cv.namedWindow("R Channel", cv.WINDOW_NORMAL)
        cv.imshow("R Channel", r_normal)

    img_normal = cv.merge([b_normal, g_normal, r_normal])
    
    # Show normalized image
    if display:
        cv.namedWindow("Normalized image", cv.WINDOW_NORMAL)
        cv.imshow("Normalized image", img_normal)

    # Compute excess green and then apply 0 threshold. Note that the
    # threshold fucntion returns the number of pixels above the
    # threshold.
    exgImg = computeIndex(img_normal, computeExG)
    
    if display:
        cv.namedWindow("ExG image", cv.WINDOW_NORMAL)
        cv.imshow("ExG image", exgImg)

    # Compute excess green minus excess red and then apply 0 threshold. Note that the
    # threshold fucntion returns the number of pixels above the
    # threshold.
    exgrImg = computeIndex(img_normal, computeExGR)
    exgrImg_thresh, exgrCount = applyThreshold(exgrImg, 0)

    if display:
        cv.namedWindow("ExGR image", cv.WINDOW_NORMAL)
        cv.imshow("ExGR image", exgrImg)
        cv.namedWindow("ExGR thresholded", cv.WINDOW_NORMAL)
        cv.imshow("ExGR thresholded", exgrImg_thresh)

    # Compute green leaf index and apply 0 threshold. Note that Rosen
    # et al. suggest using the non-normalized image, but the normalized
    # one looks to work much better.
    gliImg = computeIndex(img_normal, computeGLI)
    gliImg_thresh, gliCount = applyThreshold(gliImg, 0)

    if display:
        cv.namedWindow("GLI image", cv.WINDOW_NORMAL)
        cv.imshow("GLI image", gliImg)
        cv.namedWindow("GLI thresholded", cv.WINDOW_NORMAL)
        cv.imshow("GLI thresholded", gliImg_thresh)


    # Compute the cisible atmospherically resistant index. Again,
    # Rosen et al. suggest using the non-normalized image, but the
    # normalized one looks to work much better.
    variImg = computeIndex(img_normal, computeVARI)
    variImg_thresh, variCount = applyThreshold(variImg, 0)

    if display:
        cv.namedWindow("VARI image", cv.WINDOW_NORMAL)
        cv.imshow("VARI image", variImg)
        cv.namedWindow("VARI thresholded", cv.WINDOW_NORMAL)
        cv.imshow("VARI thresholded", variImg_thresh)
    
    if display:
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    print("There are ", exgrCount, "pixels in the ExGR image")
    print("There are ", gliCount,  "pixels in the GLI  image")
    print("There are ", variCount, "pixels in the VARI image")
    
    return 0

if __name__ == "__main__":
    main()

    
