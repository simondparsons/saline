# rgb-pick.py
#
# Code to process RGB images of plants. Initally picking pixels of
# representative regions to create a mask and then do some
# measurements.
#
# This started from my re-write of the RoI selection code from Achyut
# Paudel via:
# https://medium.com/@achyutpaudel50/hyperspectral-image-processing-in-python-custom-roi-selection-with-mouse-78fbaf7520aa
#
# and also drawing from the OpenCV colour histogram demo:
# https://github.com/opencv/opencv/blob/4.x/samples/python/tutorial_code/Histograms_Matching/histogram_calculation/calcHist_Demo.py
#
# Re-written to be a bit more interactive and do some automated blob detection.
#
# Simon Parsons
# University of Lincoln
# September 2025

# Necesary libraries
#from spectral import imshow, get_rgb
#import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#import os
#import pandas as pd
#import utils
#import csv

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

# Open the file using cv
src = cv.imread(cv.samples.findFile('./images/2024_09_02/20240902_090725.jpg'))
# Split into 
#bgr_planes = cv.split(src)

b = src[:,:,0] # get blue channel
g = src[:,:,1] # get green channel
r = src[:,:,2] # get red channel

#cv.namedWindow("B Channel", cv.WINDOW_NORMAL)
#cv.imshow("B Channel", b)
#cv.namedWindow("G Channel", cv.WINDOW_NORMAL)
#cv.imshow("G Channel", g)
#cv.namedWindow("R Channel", cv.WINDOW_NORMAL)
#cv.imshow("R Channel", r)
#cv.waitKey(0)
#cv.destroyAllWindows()

#uncorrected = envi.open('../data/raw-data-240924/linseed_b_24_09_24.hdr','../da#ta/raw-data-240924/linseed_b_24_09_24.dat')
#data_ref = envi.open('../data/raw-data-240924/linseed_b_24_09_24-gain-adjusted.hdr','../data/raw-data-240924/linseed_b_24_09_24-gain-adjusted.dat')
#bands = uncorrected.bands.centers #data_ref.bands.centers       # List of bands
#raw_data = np.array(data_ref.load()) # The raw image data

#Get an RGB image which we will use to make our selections on. We use
# the default bands.
#rgbImage = get_rgb(raw_data)

# Now view the RGB image, and set our mouse_callback function to
# record mouse clicks on the image.
#cv.namedWindow("Pick your points", cv.WINDOW_NORMAL)
# Set mouse callback function for window
#cv.setMouseCallback("Pick your points", mouse_callback)
#cv.imshow("Pick your points", src)
#cv.waitKey(0)
#cv.destroyAllWindows()

#print(mouse_clicks)

# Now extract the reflectance at each point.
#intensities = np.zeros((3, len(mouse_clicks)))
#print("Intensities: ", intensities)
#print(bgr_planes[0])
#for i in range(len(mouse_clicks)):
#    x = mouse_clicks[i][0]
#    y = mouse_clicks[i][1]
#    intensities[0][i] = b[x][y]
#    intensities[1][i] = g[x][y]
#    intensities[2][i] = r[x][y]

#    print(b[x][y])
#    print(g[x][y])
#    print(r[x][y])
    
#print("Intensities: ", intensities)
                       
# Extract the max and min values.
#thresholds = np.zeros((3, 2))

#for i in range(3):
#    max = 0
#    min = 256
#    for j in range(len(mouse_clicks)):
#        if intensities[i][j] > max:
#            max = intensities[i][j]
#        if intensities[i][j] < min:
#            min = intensities[i][j]
#    thresholds[i][0] = min
#    thresholds[i][1] = max
            
#print("Thresholds: ", thresholds)

# Now create a new image based on thresholds defined by the points
# that were selected.

newImg = np.zeros(src.shape)

#for i in range(src.shape[0]):
#    for j in range(src.shape[1]):
#        # If B, G and R values are within thresholds
#        if b[i][j] >= thresholds[0][0] and b[i][j] <= thresholds[0][1] and g[i][j] >= thresholds[1][0] and g[i][j] <= thresholds[1][1] and r[i][j] >= thresholds[2][0] and r[i][j] <= thresholds[2][1]:
        # If G values are within the thresholds
        #if bgr_planes[1][i][j] > thresholds[1][0] and bgr_planes[1][i][j] < thresholds[1][1]:
#            newImg[i][j][0] = 255
#            newImg[i][j][1] = 255
#            newImg[i][j][2] = 255

for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        # If G values are above threshold
        if g[i][j] >= 200:
            newImg[i][j][0] = 255
            newImg[i][j][1] = 255
            newImg[i][j][2] = 255
            
# Now view the resulting RGB image.
cv.namedWindow("Pixels within threshold", cv.WINDOW_NORMAL)
# Set mouse callback function for window
cv.imshow("Pixels within threshold", newImg)

newImg2 = np.zeros(src.shape)
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        # If G values are above threshold
        if g[i][j] >= 200 and r[i][j] <= 200:
            newImg2[i][j][0] = 255
            newImg2[i][j][1] = 255
            newImg2[i][j][2] = 255

# Now view the resulting RGB image.
cv.namedWindow("Pixels within threshold 2", cv.WINDOW_NORMAL)
# Set mouse callback function for window
cv.imshow("Pixels within threshold", newImg2)

cv.waitKey(0)
cv.destroyAllWindows()

exit()

# Write the set of intensities to a CSV file
# Borrowing from: https://docs.python.org/3/library/csv.html
#
with open('output.csv', 'w', newline='') as csvfile:
    iWriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    iWriter.writerow(bands)
    for intensity in intensities:
        iWriter.writerow(intensity)

# plotter.py can be used to read  this file and plot the waveforms.
