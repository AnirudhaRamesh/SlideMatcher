import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img_array = [] 
truth_array = []
truth_array.append(cv.imread('sample_test/slides/ppt1.jpg',cv.IMREAD_GRAYSCALE)) # trainImage
truth_array.append(cv.imread('sample_test/slides/ppt2.jpg',cv.IMREAD_GRAYSCALE)) 

for i in range(0,10):
    img_path = 'sample_test/frames/'+str(i)+'.jpg'
    img_array.append(cv.imread(img_path,cv.IMREAD_GRAYSCALE))      # queryImage

# Initiate SIFT detector and BFMatcher 
sift = cv.xfeatures2d.SIFT_create()
bf = cv.BFMatcher()
# find the keypoints and descriptors with SIFT
SIFT_features_array =  []  
for l in range(0,len(truth_array)):
    temp_holder, temp_features = sift.detectAndCompute(truth_array[l],None)
    SIFT_features_array.append(temp_features)

#old code below
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params

for i in range(0,10) :
    max_good_count = [-1,-1]
    kp2, des2 = sift.detectAndCompute(img_array[i],None)
    for l in range(0,len(truth_array)) :
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
            max_good_count[0] = l

    print("BEST MATCH FOR IMG", i, "IS ", max_good_count)
