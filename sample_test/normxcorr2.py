import numpy as np
import scipy
from scipy.signal import fftconvolve
import os
from os.path import isdir, join
from os import listdir
import random
import cv2


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


def main():
	
	mypath_1 = './frames/'
	files = [f for f in listdir(mypath_1)]
	files.sort()

	for i in range(0, len(files)):
		file = files[i]
		z = mypath_1 + file
		img = scipy.ndimage.imread(z, mode='L')

		mypath_2 = './slides/'
		truthfiles = [f for f in listdir(mypath_2)]
		truthfiles.sort()

		maxima = []

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
			#c = normxcorr2(ground, img)
			method = 'cv2.TM_CCORR_NORMED'
			res = cv2.matchTemplate(img,ground,eval(method))
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

			#maxc = max(map(max, c))
			#print(maxc)
			maxima.append(max_val)

		print(max(maxima))
		print(file)
		print(truthfiles[maxima.index(max(maxima))])

main()
