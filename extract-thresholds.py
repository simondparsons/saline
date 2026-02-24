# extract-thresholds.py
#
# Code to pick pixels from RGB images of plants, extract the max and
# min BGR values from the points, and then applhythose thresholds to
# create a binary mask.
#
# This started from my re-write of the RoI selection code from Achyut
# Paudel via:
# https://medium.com/@achyutpaudel50/hyperspectral-image-processing-in-python-custom-roi-selection-with-mouse-78fbaf7520aa
#
#
# Simon Parsons
# University of Lincoln
# 26-02-24

# Necesary libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Using mouse clicks requires we use a global variable. The [x, y]
# for each left-click event will be stored here
mouse_clicks =[]

# Mouse callback
#
# This function will be called at every mouse event. We want to just
# log left click locations.
#
# Left clicks are more elegant than right clicks.
def mouse_callback(event, x, y, flags, params):
    
    #the left-click event value is 1
    if event == 1:
        global mouse_clicks
        #store the coordinates of the left-click event
        mouse_clicks.append([x, y])

# This grabs an area around the mouse click.
def extract_roi(arr, x, y, w, h, intensity, line):
    roi = arr[y:y+h, x:x+w, :]
    bounding_box = arr
    #THIS PART IS JUST COLORING THE BOX AROUND THE IMAGE
    bounding_box[y-line:y, x-line:x+w+line, :] = intensity 
    bounding_box[y:y+h, x-line:x, :] = intensity 
    bounding_box[y+h:y+h+line, x-line:x+w+line, :] = intensity 
    bounding_box[y:y+h, x+w:x+w+line, :] = intensity 

    return (roi, bounding_box)

def main():
    # Allow command line specification of the input file
    parser = argparse.ArgumentParser(description='Simple threshold application for RGB images.')
    parser.add_argument('--input', help='Path to input image.')
    parser.add_argument('--debug', help='Display debug messages.')
    args = parser.parse_args()

    # Load the image file identified in the command line
    src = cv.imread(args.input)
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)

     # Do we display debug messages?
    if args.debug != None:
        debug = True
    else:
        debug = False
    
    b = src[:,:,0] # get blue channel
    g = src[:,:,1] # get green channel
    r = src[:,:,2] # get red channel

    # Now view the RGB image, and set our mouse_callback function to
    # record mouse clicks on the image.
    cv.namedWindow("Pick your points", cv.WINDOW_NORMAL)
    # Set mouse callback function for window
    cv.setMouseCallback("Pick your points", mouse_callback)
    cv.imshow("Pick your points", src)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if debug:
        print("Mouse clicks", mouse_clicks)
    
    # Now extract the reflectance at each point.
    intensities = np.zeros((3, len(mouse_clicks)))

    for i in range(len(mouse_clicks)):
        x = mouse_clicks[i][0]
        y = mouse_clicks[i][1]
        intensities[0][i] = b[x][y]
        intensities[1][i] = g[x][y]
        intensities[2][i] = r[x][y]
    
    if debug:
        print("Intensities: ", intensities)
        
    # Extract the max and min values.
    thresholds = np.zeros((3, 2))
        
    for i in range(3):
        max = 0
        min = 256
        for j in range(len(mouse_clicks)):
            if intensities[i][j] > max:
                max = intensities[i][j]
            if intensities[i][j] < min:
                min = intensities[i][j]
                
        thresholds[i][0] = min
        thresholds[i][1] = max

    if debug:
        print("Thresholds: ", thresholds)
    
    # Now create a new image based on thresholds defined by the points
    # that were selected.

    newImg = np.zeros(src.shape)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # If B, G and R values are within thresholds
            if b[i][j] >= thresholds[0][0] and b[i][j] <= thresholds[0][1] and g[i][j] >= thresholds[1][0] and g[i][j] <= thresholds[1][1] and r[i][j] >= thresholds[2][0] and r[i][j] <= thresholds[2][1]:
                newImg[i][j][0] = 255
                newImg[i][j][1] = 255
                newImg[i][j][2] = 255
            
    # Now view the resulting RGB image.
    cv.namedWindow("Threshold 1", cv.WINDOW_NORMAL)
    # Set mouse callback function for window
    cv.imshow("Threshold 1", newImg)

    newImg2 = np.zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            # If G values are above threshold
            if g[i][j] >= thresholds[1][0] + (0.5 * (thresholds[1][1] - thresholds[1][0])):
                newImg2[i][j][0] = 255
                newImg2[i][j][1] = 255
                newImg2[i][j][2] = 255

    # Now view the resulting RGB image.
    cv.namedWindow("Threshold 2", cv.WINDOW_NORMAL)
    # Set mouse callback function for window
    cv.imshow("Threshold 2", newImg2)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return 0

if __name__ == "__main__":
    main()

