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

mypath_1 = 'sample_test_2/slides/'  
onlyfiles_slides = [f for f in listdir(mypath_1) if isfile(join(mypath_1, f))]
onlyfiles_slides.sort()

mypath_2 = 'sample_test_2/frames/'  
onlyfiles_frames = [f for f in listdir(mypath_2) if isfile(join(mypath_2, f))]
onlyfiles_frames.sort()



def normxcorr2(template, image, mode="full"):

	# If this happens, it is probably a mistake

	if np.ndim(template) > np.ndim(image) or \
			len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
		#print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")
		temp = template
		template = image
		image = temp

	template = template - np.mean(template)
	image = image - np.mean(image)
	a1 = np.ones(template.shape)

	# Faster to flip up down and left right then use fftconvolve instead of scipy's correlate

	ar = np.flipud(np.fliplr(template))
	out = fftconvolve(image, ar.conj(), mode=mode)

	image = fftconvolve(np.square(image), a1, mode=mode) - \
			np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

	# Remove small machine precision errors after subtraction
	image[np.where(image < 0)] = 0
	template = np.sum(np.square(template))

	out = out / np.sqrt(image * template)

	# Remove any divisions by 0 or very close to 0
	out[np.where(np.logical_not(np.isfinite(out)))] = 0

	return out


def xcorrcheck(file, best_slides):
    mypath_1 = './sample_test_2/frames/'
    
    z = mypath_1 + file
    img = scipy.ndimage.imread(z, mode='L')

    mypath_2 = './sample_test_2/slides/'

    maxima = []
    truthfiles = []

    for i in range(0, len(best_slides)):
        truthfiles.append(best_slides[i][0])

    for j in range(0, len(truthfiles)):
        w = mypath_2 + truthfiles[j]
        ground = scipy.ndimage.imread(w, mode='L')

        temp = img

        if np.ndim(ground) > np.ndim(img) or \
            len([i for i in range(np.ndim(ground)) if ground.shape[i] > img.shape[i]]) > 0:
            #print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")
            temp = ground
            ground = img

        #c = scipy.signal.correlate2d(temp, ground)
        c = normxcorr2(ground, temp)
        #method = 'cv2.TM_CCORR_NORMED'
        #res = cv2.matchTemplate(img,ground,eval(method))
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        maxc = max(map(max, c))
        print(maxc)
        maxima.append(maxc)

    #print(max(maxima))
    print("Inside normxcorr best result for", file," = ",truthfiles[maxima.index(max(maxima))])
    
    output = []
    output.append(truthfiles[maxima.index(max(maxima))])
    max_index = maxima.index(max(maxima))
    maxima[max_index] = 0
    output.append(truthfiles[maxima.index(max(maxima))])
    max_index = maxima.index(max(maxima))
    maxima[max_index] = 0
    output.append(truthfiles[maxima.index(max(maxima))])
    max_index = maxima.index(max(maxima))
    maxima[max_index] = 0
    output.append(truthfiles[maxima.index(max(maxima))])
    max_index = maxima.index(max(maxima))
    maxima[max_index] = 0
    output.append(truthfiles[maxima.index(max(maxima))])
    max_index = maxima.index(max(maxima))
    maxima[max_index] = 0

    print(output)

    answer = SSIM(img, output)
    return answer

def check(i, max_good_count) :
    """ Function to check internally of accuracy """ 
    folder_number = onlyfiles_frames[i] ; 
    folder_number = folder_number.split('_')
    folder_number = folder_number[0]
    answered_number =  max_good_count
    answered_number =  answered_number.split('-')
    answered_number = answered_number[1]
    answered_number = answered_number.split('.')
    answered_number = answered_number[0]
    print('ayy lmao' + answered_number)
    print(folder_number) 
    global total_correct
    if folder_number == answered_number : 
        total_correct += 1 

    return 

def SSIM(img, output):
    
    temp_ssim = []

    for j in range(0, len(output)):
        mypath_1 = './sample_test_2/slides/'   
        z = mypath_1 + output[j]
        print(z)
        ground = scipy.ndimage.imread(z, mode='L')

        if np.ndim(ground) != np.ndim(img) or \
        len([k for k in range(np.ndim(ground)) if ground.shape[k] != img.shape[k]]) > 0:
        #print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")
        #temp = ground
        #ground = img
            width = int(img.shape[1])
            height = int(img.shape[0])
            dim = (width, height) 
            ground = cv.resize(ground, dim) 
            print('Image has been resized')

        temp_ssim.append(ssim(img, ground))

    max_ind = temp_ssim.index(max(temp_ssim))   
    # print(onlyfiles_frames[i])
    print(output[max_ind])

    return output[max_ind]


def SIFT():
    sift = cv.xfeatures2d.SIFT_create()
    bf = cv.BFMatcher()
    # find the keypoints and descriptors with SIFT
    SIFT_features_array =  []  
    SIFT_plotting_array = [] 

    for l in range(0,truth_array_len):
        temp_holder, temp_features = sift.detectAndCompute(truth_array[l],None)
        SIFT_features_array.append(temp_features)
        SIFT_plotting_array.append(temp_holder)

    #old code below
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params

    for i in range(45,70) :
        max_good_count = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
        kp2, des2 = sift.detectAndCompute(img_array[i],None)
        for l in range(0,truth_array_len) :
            temp = SIFT_features_array[l]
            matches = bf.knnMatch(temp,des2,k=2)
            # Apply ratio test
            good = []
            good_count = 0 
            for m,n in matches:
                if m.distance < 0.5*n.distance:
                    good.append([m])
                    good_count += 1
                    # print(onlyfiles_frames[i], max_good_count)
    # cv.drawMatchesKnn expects list of lists as matches.

            # if onlyfiles_slides[l] == 'ppt-100.jpg' or  onlyfiles_slides[l] == 'ppt-101.jpg' : 
            #     img3 = cv.drawMatchesKnn(truth_array[l],SIFT_plotting_array[l],img_array[i],kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #     plt.imshow(img3),plt.show()
            
            # print(onlyfiles_slides[l], good_count)


            # if onlyfiles_slides[l] == 'ppt-13.jpg':
            #     print(good_count)
                
            if max_good_count[0][1] < good_count :
                max_good_count[0][1] = good_count
                max_good_count[0][0] = onlyfiles_slides[l]
            elif max_good_count[1][1] < good_count :
                max_good_count[1][1] = good_count
                max_good_count[1][0] = onlyfiles_slides[l]
            elif max_good_count[2][1] < good_count :
                max_good_count[2][1] = good_count
                max_good_count[2][0] = onlyfiles_slides[l]
            elif max_good_count[3][1] < good_count :
                max_good_count[3][1] = good_count
                max_good_count[3][0] = onlyfiles_slides[l]
            elif max_good_count[4][1] < good_count :
                max_good_count[4][1] = good_count
                max_good_count[4][0] = onlyfiles_slides[l]
            elif max_good_count[5][1] < good_count :
                max_good_count[5][1] = good_count
                max_good_count[5][0] = onlyfiles_slides[l]
            elif max_good_count[6][1] < good_count :
                max_good_count[6][1] = good_count
                max_good_count[6][0] = onlyfiles_slides[l]
            elif max_good_count[7][1] < good_count :
                max_good_count[7][1] = good_count
                max_good_count[7][0] = onlyfiles_slides[l]
            elif max_good_count[8][1] < good_count :
                max_good_count[8][1] = good_count
                max_good_count[8][0] = onlyfiles_slides[l]
            elif max_good_count[9][1] < good_count :
                max_good_count[9][1] = good_count
                max_good_count[9][0] = onlyfiles_slides[l]

            max_good_count.sort(key=lambda tup:tup[1])
            
                

        # if onlyfiles_slides[l] == 'ppt-13.jpg':
        #     max_good_count[0][1] = good_count
        #     max_good_count[0][0] = onlyfiles_slides[l]
        #     print(good_count)

        print(onlyfiles_frames[i], max_good_count[0])
        print(onlyfiles_frames[i], max_good_count[1])
        print(onlyfiles_frames[i], max_good_count[2])
        print(onlyfiles_frames[i], max_good_count[3])
        print(onlyfiles_frames[i], max_good_count[4])
        print(onlyfiles_frames[i], max_good_count[5])
        print(onlyfiles_frames[i], max_good_count[6])
        print(onlyfiles_frames[i], max_good_count[7])
        print(onlyfiles_frames[i], max_good_count[8])
        print(onlyfiles_frames[i], max_good_count[9])
        
        best_match = xcorrcheck(onlyfiles_frames[i], max_good_count) 
        # For testing purposes here 
        print("FINAL ANSWER FOR ", onlyfiles_frames[i], best_match )
        check(i, best_match)

    return 

# def cross_corelation():

#     for i in range(0,img_array_len) :
#         min_var = [ -1, 1000000000000000000]
#         for l in range(0,truth_array_len) :
#             var = sqrt(sum(sum((img_array[l])*(truth_array[l]))))
#             if min_var[1] < var  :
#                 min_var[0] = onlyfiles_slides[l]
#                 min_var[1] = var 

#         print(onlyfiles_frames[i], min_var)

#         check(i, min_var)

            
            



for i in onlyfiles_slides:
    file_name = mypath_1 + i 
    truth_array.append(cv.imread(file_name,cv.IMREAD_GRAYSCALE)) # trainImage

truth_array_len = len(truth_array)

for i in onlyfiles_frames:
    file_name = mypath_2 + i 
    img_array.append(cv.imread(file_name,cv.IMREAD_GRAYSCALE))      # queryImage

img_array_len = len(img_array)

#SSIM()
SIFT() 
#cross_corelation()

# Initiate SIFT detector and BFMatcher 
print("Accuracy = ", (total_correct*100)/25)


