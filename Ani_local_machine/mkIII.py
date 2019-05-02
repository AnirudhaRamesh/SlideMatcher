import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from math import sqrt 
import scipy
from scipy.signal import fftconvolve
import os
from os.path import isdir
import random
from skimage.measure import compare_ssim as ssim 
from skimage import io, feature
from scipy import ndimage


img_array = [] 
truth_array = []
total_correct = 0 

mypath_1 = 'sample_test/slides/'  
onlyfiles_slides = [f for f in listdir(mypath_1) if isfile(join(mypath_1, f))]
onlyfiles_slides.sort()

mypath_2 = 'sample_test/frames/'  
onlyfiles_frames = [f for f in listdir(mypath_2) if isfile(join(mypath_2, f))]
onlyfiles_frames.sort()


def cross_covariance():

    random_test_image_selection = np.random.randint(0,834,50)
    method = eval('cv.TM_CCOEFF')

    for i in random_test_image_selection:
        best = [] 
        for l in range(0,truth_array_len):
            res = cv.matchTemplate(img_array[i],truth_array[l],method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            best.append([max_val, onlyfiles_frames[i], onlyfiles_slides[l]])

        max_result = max(best)
        print(max_result)
        check(i, max_result[2])

    return 



for i in onlyfiles_slides:
    file_name = mypath_1 + i 
    truth_array.append(cv.imread(file_name,cv.IMREAD_GRAYSCALE)) # trainImage

truth_array_len = len(truth_array)

for i in onlyfiles_frames:
    file_name = mypath_2 + i 
    img_array.append(cv.imread(file_name,cv.IMREAD_GRAYSCALE))      # queryImage

img_array_len = len(img_array)

#SSIM()
print(img_array_len)
cross_covariance()
#SIFT() 
#cross_corelation()

# Initiate SIFT detector and BFMatcher 
print("Accuracy = ", (total_correct*100)/50)


