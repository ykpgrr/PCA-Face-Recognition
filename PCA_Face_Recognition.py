'''
PCA_FACE.py
Created by Yakup Gorur (040130052) on 01 May 2018.
Copyright © 2018 Yakup Gorur. All rights reserved.

Digital Signal Processing Design and Application 2017-2018 Spring
Homework 7
Lecturer: Prof. Dr. Bilge Gunsel, Research Assistant Yağmur Sabucu

'''

import os
import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'''__________________________USER SELECTED AREA___________________________'''

# Directory containing images
dirNameTrain = 'INPUT/TRAIN'
dirNameTest	 = 'INPUT/TEST'
OUT_Path     = 'OUT/'
reconstruct_Path = 'OUT/Reconstruct_All_eigen/'
reconstruct_K_Path ='OUT/Reconstruct_K_eigen/'

# Ratio, Selecting best eigen vectors
S1_ratio_S	 = 80

'''__________________________USER SELECTED AREA___________________________'''

# Select best K eigenValues
def Specify_K_eigenvalues(eigenValues, sum_eigenValues, percent):

	for i in range(0, len(eigenValues)):
		if ( ( np.sum(eigenValues[:i]) / sum_eigenValues ) * 100 ) > percent :
			return i

	return None

# Convert the face vector to images and then Write the images
def WriteImages(facevectors, size, path):

	for i in range(0, facevectors.shape[1]):
		image = facevectors[:,i].reshape(size)
		image = image.astype(int)
		cv2.imwrite(path + 'Resim' + str(i) + '.jpg', image)

	return True

# Read a txt file as matrix
def ReadFromFile(filename, string):
	print('reading -' + str(string) +  '- from' + str(filename), end=' ... ', flush=True)
	file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
	matrix = file.getNode(string).mat()
	file.release()
	print('DONE')
	return matrix

# Save a matrix as txt
def WritetoFile(filename, string, matrix):
	print('Writing -' + str(string) +  '- to ' + filename, end=' ... ', flush=True)
	file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
	file.write(str(string), matrix)
	file.release()
	print('DONE')

# Find eigenValues and eigenVectors of a Matrix
def Find_Eigen(matrix) :
	eigenVectors , eigenValues , _ = np.linalg.svd(matrix, full_matrices = False )
	sort = eigenValues.argsort()[::-1]
	eigenValues = eigenValues[sort]
	eigenVectors = eigenVectors[:,sort]
	return eigenValues,eigenVectors

# Create Data Matrix which keeping the images as a single column
def CreateDataMatrix(images):
	print('Creating data matrix',end=' ... ', flush=True)

	numImages = len(images)
	sz = images[0].shape
	data = np.zeros(( sz[0] * sz[1], numImages), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()

		data[:,i] = image

	print('DONE')
	return data

# Read all of the Images in a folder
def ReadImages(path):
	print('Reading images from ' + path, end=' ... ', flush=True)
	# Create array of array of images.
	images = []
	# List all files in the directory and read points from text files one by one
	for filePath in sorted(os.listdir(path)):
		fileExt = os.path.splitext(filePath)[1]
		if fileExt in ['.jpg', '.jpeg', '.pgm']:
			# Add to array of images
			imagePath = os.path.join(path, filePath)
			im = cv2.imread(imagePath)
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

			if im is None :
				print('image:{} not read properly'.format(imagePath))
			else :
				# Convert image to floating point
				im = np.float32(im)
				# Add image to list
				images.append(im)
				# Flip image
				#imFlip = cv2.flip(im, 1);
				# Append flipped image
				#images.append(imFlip)

	numImages = len(images) / 2
	# Exit if no image found
	if numImages == 0 :
		print('No images found')
		sys.exit(0)

	print(str(numImages) + ' files read.')

	return images

if __name__ == '__main__':

	''' -------------------------------a-------------------------------------
	Read N face images from the training set and obtain the covariance matrix C.
	'''
	# Read images
	train_images = ReadImages(dirNameTrain)

	# Create face matrix for PCA.
	face_vector = CreateDataMatrix(train_images)

	# Size of images
	sz = train_images[0].shape
	number_train_images= face_vector.shape[1]

	# Find average_face_vector, sum(all image vectors)/number(images).
	average_face_vector = np.mean(face_vector, axis=1)
	average_face_vector.shape = (len(average_face_vector), 1)

	# Subtract average_face_vector from every image vector.
	sub_face_vector = np.zeros(face_vector.shape, dtype=np.float32)
	sub_face_vector = face_vector - average_face_vector

	# Calculate covariance matrix of above matrix -> C = A*transpose(A)
	covariance_matrix = np.dot(sub_face_vector.T, sub_face_vector)
	covariance_matrix /= number_train_images

	''' -------------------------------b and c----------------------------------
	b-) Calculate eigen values of C and order them by using a sorting algorithm.
	Report the details of your sorting algorithm.Plot the ordered E eigen values

	c-) Calculate eigen vectors corresponding to E eigen values and
	write these vectors into a file named as EV.txt.
	'''

	# Find eigenvectors and eigenvalues of above covariance matrix.
	# eigenvalues arranged to match with associated eigenvector
	print('Calculating PCA ', end=' ... ', flush=True)
	eigenValues, eigenVectors = Find_Eigen(sub_face_vector)
	print ('DONE')
	print("eigenvector", eigenVectors.shape )

	# Plot the ordered eigen values.
	fig = plt.figure()
	ax1 = plt.axes()
	ax1.plot(eigenValues)
	plt.title('eigenValues Ordered')
	plt.ylabel('Value of eigenValues')
	plt.xlabel('index number')
	plt.show()

	#Write the eigen values
	WritetoFile(OUT_Path + 'EV.txt','eigenVectors', eigenVectors)

	''' -------------------------------d---------------------------------------
	d-) Transform the used N face images into a new vector space V
	by using all of the eigenvectors (Use EV.txt).

		Write the weights into Weights.txt.
	'''
	# Calculate weights
	weights = np.dot(eigenVectors.T, sub_face_vector)

	# Write weights
	WritetoFile(OUT_Path + 'Weights.txt','Weights', weights)

	''' -------------------------------e---------------------------------------
	e-) Reconstruct N face images as a linear combination of bases of the 'V'
	(use EV.txt and Weights.txt). Write the reconstructed images as jpg files.
	'''
	plt.imshow
	# Reconstruct from 'V' Space
	reconstructed_image_vector = average_face_vector + np.dot(eigenVectors, weights)

	# Write Reconstructed Images
	WriteImages(reconstructed_image_vector, sz, reconstruct_Path)


	''' -------------------------------f---------------------------------------
	f-) Calculate the difference images between
	the N original face images and the reconstructed images.

		Calculate the mean squared error and report the reconstruction error RE1.
	'''
	#Mean Square error between all original and reconstructed images
	MSE = np.sum( np.square(face_vector - reconstructed_image_vector) ) / (face_vector.shape[1] * face_vector.shape[0] )
	print('Mean Squared Error of the reconstructed face RE1: ' + str(MSE))

	''' -------------------------------g---------------------------------------
	g-) Report sum of E eigen values as S.
		Specify K, the number of important eigen values by checking the ratio of (S1/S)
	where S1 denotes sum of the used eigen values.
		Write the selected eigen vectors into a file named as EVK.txt.
	'''

	# sum of eigen values
	print('Calculating the sum of the eigen values', end=' ... ', flush=True)
	sum_eigenValues = np.sum(eigenValues)
	print('DONE')
	print('sum of the eigen values = ' + str(sum_eigenValues))

	# Calculate the important eigen values
	print('Calculating the number of important eigen values for S1/S = %' + str(S1_ratio_S), end=' ... ', flush=True)
	K = Specify_K_eigenvalues(eigenValues, sum_eigenValues, S1_ratio_S)
	print('DONE')
	print('total number of eigen values = {} the number of important eigen values K= {}'.format(len(eigenValues), K) )

	# Select important eigen values
	selected_eigenVectors = eigenVectors[:,0:K]

	# Writing the selected eigen vectors
	WritetoFile(OUT_Path + 'EVK.txt','selected_eigenvalues', selected_eigenVectors )

	''' -------------------------------h---------------------------------------
	h-) Transform the used N face images into a new vector space VK
	by using selected K eigen vectors (Use EVK.txt).

		Write the weights into WeightsK.txt.
	'''
	# Transform the Images in  VK space using K eigenVectors
	weightsK = np.dot(selected_eigenVectors.T, sub_face_vector)

	# Writes the weights
	WritetoFile(OUT_Path + 'WeightsK.txt','WeightsK', weightsK)

	''' -------------------------------i---------------------------------------
	i-) Repeat steps (e) and (f) for the new vector space VK and
	report the reconstruction error RE2. (use EVK.txt and WeightsK.txt).
	'''

	reconstructed_image_vectorK = average_face_vector + np.dot(selected_eigenVectors, weightsK)
	WriteImages(reconstructed_image_vectorK, sz, reconstruct_K_Path)
	MSE_K = np.sum( np.square(face_vector - reconstructed_image_vectorK) ) / (face_vector.shape[1] * face_vector.shape[0] )
	print('Mean Squared Error of the reconstructed face RE2: ' + str(MSE_K))

	''' -------------------------------j---------------------------------------
	j-) j. Compare RE1 and RE2. Are they equal or not? Why?
	'''
	''' -------------------------------k---------------------------------------
	k-) Read T face images from the test set and
	transform them into the vector space V by using EV.txt.

		Write the weights into WeightsT.txt.
	'''

	# Read test images
	test_images = ReadImages(dirNameTest)

	# Create face matrix for PCA.
	test_face_vector = CreateDataMatrix(test_images)

	# Find average_face_vector, sum(all image vectors)/number(images).
	average_test_face_vector = test_face_vector - average_face_vector

	#Transform the test images to 'V' space
	weights_test = np.dot(eigenVectors.T, average_test_face_vector)

	#Write Test Weights
	WritetoFile(OUT_Path + 'WeightsT.txt','WeightsT', weights_test)

	''' -------------------------------L---------------------------------------
	L-) Calculate the Euclidean distance between each of T test images and N training images.
		Match the test images to the one training image that minimizes the Euclidean distance
		(Use Weights.txt and WeightsT.txt).
		Count the number of correctly matched faces and report it as TP.

	'''

	# Create array to keep matched images
	Matched_Faces = np.zeros((weights_test.shape[1]))

	# Compare all weights_test among the weights and Calculate Similarity
	for i in range(0, weights_test.shape[1]):
		error = np.zeros((weights.shape[1]))
		for j in range (0, weights.shape[1]):
			error[j] = (np.sum((weights[:, j] - weights_test[:, i])**2))

		# Match the test image with training image
		Matched_Faces[i] = error.argmin() // 7

	# Print the indexes of matched images
	print(Matched_Faces)

	''' -------------------------------M---------------------------------------
	M-) Repeat (k) and (i) by using EVK.txt.
		Write the weights into WeightsTK.txt.
		Apply matching by using WeightsK.txt and WeightsTK.txt.
		Count the number of correctly matched faces and report it as TPK.

	'''
	# Read test images
	test_images_K = ReadImages(dirNameTest)

	# Create face matrix for PCA.
	test_face_vector_K = CreateDataMatrix(test_images_K)

	# Find average_face_vector, sum(all image vectors)/number(images).
	average_test_face_vector_K = test_face_vector_K - average_face_vector

	#Transform the test images to 'VK' space
	weights_test_K = np.dot(selected_eigenVectors.T, average_test_face_vector_K)

	#Write Test Weights K
	WritetoFile(OUT_Path + 'WeightsTK.txt','WeightsTK', weights_test_K)

	# Create array to keep matched images
	Matched_Faces_K = np.zeros((weights_test_K.shape[1]))

	# Compare all weights_test_K among the weights_K and Calculate Similarity
	for i in range(0, weights_test_K.shape[1]):
		error_K = np.zeros((weights.shape[1]))
		for j in range (0, weights.shape[1]):
			AAA = np.pad(weights_test_K[:, i], (0, weights.shape[0] - weights_test_K.shape[0]), 'constant')
			error_K[j] = (np.sum((weights[:, j] - AAA)**2))

		# Match the test image with training image
		Matched_Faces_K[i] = error_K.argmin() // 7

	# Print the indexes of matched images
	print(Matched_Faces_K)

	''' -------------------------------O---------------------------------------
	O-) Repeat (k) and (i) by using face images recorded in the training set as test images.
	Report the number of correctly matched images as TPT.
	Compare TPT and TP and comments on your face verification accuracy.
	'''

	# We already have training weights in 'weights'

	# Create array to keep matched images
	Matched_Faces_training = np.zeros((weights.shape[1]))

	for i in range(0, weights.shape[1]):
		error_K = np.zeros((weights.shape[1]))
		for j in range (0, weights.shape[1]):
			AAA = np.pad(weights[:, i], (0, weights.shape[0] - weights.shape[0]), 'constant')
			error_K[j] = (np.sum((weights[:, j] - AAA)**2))

		# Match the test image with training image
		Matched_Faces_training[i] = error_K.argmin() // 7

	# Print the indexes of matched images
	print(Matched_Faces_training)
