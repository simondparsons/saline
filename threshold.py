# threshold.py
#
# Code to apply a simple threshold to RGB images. We're looking for
# green pixels, so something pretty straightforward seems an
# appropriate place to start.
#
# Simon Parsons
# University of Lincoln
# 26-02-23

# Necesary libraries
import argparse
import numpy as np
#import matplotlib.pyplot as plt
import cv2 as cv

# Code take from rgb-pick.py, so will look familiar

def main():
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

    # Doe we display images?
    if args.display != None:
        display = True
    else:
        display = False

    # Split in to B, G and R channels. Could use bgr_planes but the internet says this is more efficient.
    b = src[:,:,0] # get blue channel
    g = src[:,:,1] # get green channel
    r = src[:,:,2] # get red channel

    # Something to create the output mask in, and a variable to count the "green" pixels in
    newImg = np.zeros(src.shape)
    pixelCount = 0
    
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # If G values are above threshold
            if g[i][j] >= 200:
                newImg[i][j][0] = 255
                newImg[i][j][1] = 255
                newImg[i][j][2] = 255
                pixelCount = pixelCount + 1
        
    # Try adding a red threshold as well. 
    newImg2 = np.zeros(src.shape)
    pixelCount2 = 0
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # If G values are above threshold and R values are below
            if g[i][j] >= 200 and r[i][j] <= 175:
                newImg2[i][j][0] = 255
                newImg2[i][j][1] = 255
                newImg2[i][j][2] = 255
                pixelCount2 = pixelCount2 + 1

    # Now view the resulting images if requested
    if display:
        # Display windows and set callback
        cv.namedWindow("First threshold", cv.WINDOW_NORMAL)
        cv.imshow("First threshold", newImg)

        cv.namedWindow("Second threshold", cv.WINDOW_NORMAL)
        cv.imshow("Second threshold", newImg2)
    
        cv.waitKey(0)
        cv.destroyAllWindows()

    
    print("There are ", pixelCount, "green pixels")
    print("There are ", pixelCount2, "red/green pixels")
    
    return 0

if __name__ == "__main__":
    main()
