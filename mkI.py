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
img = cv.imread('contrast.png')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img = cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)