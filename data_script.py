import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


f = listdir('Dataset')
all_folders = []
slides_data = []
camera_data = []
camera_data_content = []
slides_data_content = []

# camera_data_kp = []
# slides_data_kp = []

# camera_data_des = []
# sildes_data_des = []

for i in f:
	all_folders.append("Dataset/"+i)

for i in all_folders:
	temp_folder = listdir(i)
	for j in temp_folder:
		if j == 'ppt.jpg':
			slides_data.append(i + '/' + j)
		else:
			camera_data.append(i + '/' + j)


#Gonna just store the urls rather than contents themselves
# for i in camera_data:
# 	camera_data_content.append(cv.imread(i,cv.IMREAD_GRAYSCALE))


# for i in slides_data:
# 	slides_data_content.append(cv.imread(i,cv.IMREAD_GRAYSCALE))

#some initialisation for flann stuff
sift = cv.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)

ans_checking = []

for i in camera_data:
	print("\nNow checking for image in:"+i)
	max_value = 0 #useless comment, but we'll be maximising this
	img = cv.imread(i,cv.IMREAD_GRAYSCALE)
	len_h_1 = len(img)
	wid_h_1 = len(img[0])
	print(len_h_1)
	print(wid_h_1)
	img = np.array(img)
	kp1, des1 = sift.detectAndCompute(img, None)

	# img_1 = img[0:len_h_1//2][0:wid_h_1//2]
	#  # = img[0:len_h_1//2][wid_h_1//2:wid_h_1]
	# img_2 = [img[i][wid_h_1//2:wid_h_1] for i in range(len_h_1//2)]
	# img_2 = np.array(img_2)

	# img_3 = img[len_h_1//2:len_h_1][0:wid_h_1//2]
	# img_4 = [img[i][wid_h_1//2:wid_h_1] for i in range(len_h_1//2,len_h_1)]
	# img_4 = np.array(img_4)

	# kp1_1, des1_1 = sift.detectAndCompute(img_1, None)
	# kp1_2, des1_2 = sift.detectAndCompute(img_2, None)
	# kp1_3, des1_3 = sift.detectAndCompute(img_3, None)
	# kp1_4, des1_4 = sift.detectAndCompute(img_4, None)

	itr = 0
	index = 0

	for j in slides_data:
		print("comparing with : " + j)
		itr += 1
		img = cv.imread(j, cv.IMREAD_GRAYSCALE)
		len_h = len(img)
		wid_h = len(img[0])
		if len_h == len_h_1 and wid_h == wid_h_1:
			img = np.array(img)
			kp2, des2 = sift.detectAndCompute(img, None)
			matches = flann.knnMatch(des1,des2,k=2)

			# img_1 = img[0:len_h//2][0:wid_h//2]
			# img_2 = [img[i][wid_h//2:wid_h] for i in range(len_h//2)]
			# img_2 = np.array(img_2)
			# img_3 = img[len_h//2:len_h][0:wid_h//2]
			# img_4 = [img[i][wid_h//2:wid_h] for i in range(len_h//2,len_h)]
			# img_4 = np.array(img_4)

			# kp2_1, des2_1 = sift.detectAndCompute(img_1, None)
			# kp2_2, des2_2 = sift.detectAndCompute(img_2, None)
			# kp2_3, des2_3 = sift.detectAndCompute(img_3, None)
			# kp2_4, des2_4 = sift.detectAndCompute(img_4, None)

			# if type(des2_1) != None and type(des1_1)!=None:
			# 	matches_1 = flann.knnMatch(des1_1,des2_1,k=2)
			# if type(des2_2) != None and type(des1_2)!=None:
			# 	matches_2 = flann.knnMatch(des1_2,des2_2,k=2)
			# if type(des2_3) != None and type(des1_3)!=None:
			# 	matches_3 = flann.knnMatch(des1_3,des2_3,k=2)
			# if type(des2_4) != None and type(des1_4)!=None:
			# 	matches_4 = flann.knnMatch(des1_4,des2_4,k=2)

			good = []
			for i,(m,n) in enumerate(matches):
				if m.distance < 0.5*n.distance:
			 		good.append([m])

			# for m,n in matches_2:
			# 	if m.distance < 0.75*n.distance:
			# 		good.append([m])

			# for m,n in matches_3:
			# 	if m.distance < 0.75*n.distance:
			# 		good.append([m])

			# for m,n in matches_4:
			# 	if m.distance < 0.75*n.distance:
			# 		good.append([m])

			if len(good) > max_value:
				max_value = len(good)
				index = itr
		print("Max value is : ")
		print(max_value)		
		# img = np.array(img)
		# img_1 = img[0:len_h//2][0:wid_h//2]
		# # = img[0:len_h//2][wid_h//2:wid_h]
		# img_2 = [img[i][wid_h//2:wid_h] for i in range(len_h//2)]
		# img_2 = np.array(img_2)
		# img_3 = img[len_h//2:len_h][0:wid_h//2]
		# img_4 = [img[i][wid_h//2:wid_h] for i in range(len_h//2,len_h)]
		# img_4 = np.array(img_4)

		# kp2_1, des2_1 = sift.detectAndCompute(img_1, None)
		# kp2_2, des2_2 = sift.detectAndCompute(img_2, None)
		# kp2_3, des2_3 = sift.detectAndCompute(img_3, None)
		# kp2_4, des2_4 = sift.detectAndCompute(img_4, None)

		# if type(des2_1) != None and type(des1_1)!=None:
		# 	matches_1 = flann.knnMatch(des1_1,des2_1,k=2)
		# if type(des2_2) != None and type(des1_2)!=None:
		# 	matches_2 = flann.knnMatch(des1_2,des2_2,k=2)
		# if type(des2_3) != None and type(des1_3)!=None:
		# 	matches_3 = flann.knnMatch(des1_3,des2_3,k=2)
		# if type(des2_4) != None and type(des1_4)!=None:
		# 	matches_4 = flann.knnMatch(des1_4,des2_4,k=2)
		
		# good = []

		# for m,n in matches_1:
		# 	if m.distance < 0.75*n.distance:
		# 		good.append([m])

		# for m,n in matches_2:
		# 	if m.distance < 0.75*n.distance:
		# 		good.append([m])

		# for m,n in matches_3:
		# 	if m.distance < 0.75*n.distance:
		# 		good.append([m])

		# for m,n in matches_4:
		# 	if m.distance < 0.75*n.distance:
		# 		good.append([m])

		# if len(good) > max_value:
		# 	max_value = len(good)
		# 	index = itr

	ans_checking.append(index)
	print("Best fit for this is in" + slides_data[index])







#old code below
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# flannMatcher with default params

# img1 = cv.imread('0.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('ppt.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# # Initiate SIFT detector
# sift = cv.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1, des2,k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])