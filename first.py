# A first attempt to use the PlantCV library.

# Based on:
# https://plantcv.readthedocs.io/en/stable/tutorials/vis_tutorial/

import cv2 as cv
#from plantcv import plantcv as pcv
import matplotlib as matplotlib
from matplotlib import pyplot as plt
#from plantcv.parallel import WorkflowInputs

matplotlib.use('TkAgg')

#args = WorkflowInputs(
#    images=["./images/2024_09_02/20240902_085316.jpg"],
#    names="image",
#    result="example_results_oneimage_file.json",
#    outdir=".",
#    writeimg=False,
#    debug="plot"
#    )

# Read in image
#img, path, filename = pcv.readimage(filename=args.image)
image  = cv.imread("./images/2024_09_09/20240909_082448.jpg")
image2 = cv.imread("./images/2024_09_09/20240909_082502.jpg")
image3 = cv.imread("./images/2024_09_02/20240902_085316.jpg")
image4 = cv.imread("./images/2024_05_13/20240513_115823.jpg")


#print('Datatype:', img.dtype, '\nDimensions:', img.shape)
print('Datatype:', image.dtype, '\nDimensions:', image.shape)

# Now some stuff from
# https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
# to compute colour histograms:
color = ('b','g','r')
plt.subplot(4, 1, 1)
for i,col in enumerate(color):
    histr = cv.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

print(cv.calcHist([image],[1],None,[256],[0,256]))
      
plt.subplot(4, 1, 2)
for i,col in enumerate(color):
    histr = cv.calcHist([image2],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.subplot(4, 1, 3)
for i,col in enumerate(color):
    histr = cv.calcHist([image3],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.subplot(4, 1, 4)
for i,col in enumerate(color):
    histr = cv.calcHist([image4],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    
plt.show()

exit()

# Use OpenCV to display the image, resizing since the base size
# exceeds my screen.
# This is based on:
# https://learnopencv.com/image-resizing-with-opencv/
# Get initial size:
height, width, channels = image.shape
scaling_factor = 0.2
down_width = int(width * scaling_factor)
down_height = int(height * scaling_factor)
down_points = (down_width, down_height)
resized_down = cv.resize(image, down_points, interpolation= cv.INTER_LINEAR)
window_name = 'image'
cv.imshow(window_name, resized_down)
cv.waitKey(0)
cv.destroyAllWindows()
