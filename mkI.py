# import numpy as numpy
# import cv2
# from matplotlib import pyplot as plt

# img1 = cv2.imread('contrast.png', 0)
# img2 = cv2.imread('contrast-truth.png', 0)

# orb = cv2.ORB_create()

# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# matches = bf.match(des1,des2)

# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# # Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, None, flags=2)

# plt.imshow(img3),plt.show()
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('0.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('ppt.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()