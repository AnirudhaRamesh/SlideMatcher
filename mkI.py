import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


img_array = [] 
truth_array = []
total_correct = 0 

mypath_1 = 'sample_test/slides/'  
onlyfiles_slides = [f for f in listdir(mypath_1) if isfile(join(mypath_1, f))]
onlyfiles_slides.sort()

mypath_2 = 'sample_test/frames/'  
onlyfiles_frames = [f for f in listdir(mypath_2) if isfile(join(mypath_2, f))]
onlyfiles_frames.sort()

for i in onlyfiles_slides:
    file_name = mypath_1 + i 
    truth_array.append(cv.imread(file_name,cv.IMREAD_GRAYSCALE)) # trainImage

truth_array_len = len(truth_array)

for i in onlyfiles_frames:
    file_name = mypath_2 + i 
    img_array.append(cv.imread(file_name,cv.IMREAD_GRAYSCALE))      # queryImage

img_array_len = len(img_array)

# Initiate SIFT detector and BFMatcher 
sift = cv.xfeatures2d.SIFT_create()
bf = cv.BFMatcher()
# find the keypoints and descriptors with SIFT
SIFT_features_array =  []  
for l in range(0,truth_array_len):
    temp_holder, temp_features = sift.detectAndCompute(truth_array[l],None)
    SIFT_features_array.append(temp_features)

#old code below
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params

for i in range(0,img_array_len) :
    max_good_count = [-1,-1]
    kp2, des2 = sift.detectAndCompute(img_array[i],None)
    for l in range(0,truth_array_len) :
        temp = SIFT_features_array[l]
        matches = bf.knnMatch(temp,des2,k=2)
        # Apply ratio test
        good = []
        good_count = 0 
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
                good_count += 1
# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()
        if max_good_count[1] < good_count :
            max_good_count[1] = good_count
            max_good_count[0] = onlyfiles_slides[l]

    print(onlyfiles_frames[i], max_good_count)

    # For testing purposes here 
    check(i, max_good_count)

print("Accuracy = ", (total_correct*100)/img_array_len)



def check(i, max_good_count) :
    """ Function to check internally of accuracy """ 
    folder_number = onlyfiles_frames[i][0] ; 
    answered_number =  max_good_count[0][4] ; 
    if folder_number == answered_number : 
        total_correct += 1 

    return 


