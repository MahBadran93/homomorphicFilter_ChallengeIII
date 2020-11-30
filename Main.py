import logging
import numpy as np
import cv2
import  matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

#..........................High Pass Filters...............................................

def Gaussian(inputImage, segma = 30):
    # find center of the image 
    P = inputImage.shape[0]/2
    Q = inputImage.shape[1]/2

    # to vectorise the operation we use mesh grid.
    width = np.arange(inputImage.shape[0]) # give us a numpy array from 1 -> width
    height = np.arange(inputImage.shape[1]) # give us a numpy array from 1 -> height
    array1, array2 = np.meshgrid(width, height,sparse=False, indexing='ij') # array1: array of row index values, array2: array of column index values

    # Gaussian image
    nom = np.square(array1 - P) + np.square(array2 - Q)
    expVal = -nom / (2 * (segma**2))
    gaussianArray = 1 - np.exp(expVal)
    return gaussianArray

 

'''
def Butterworth( inputImage, sigma = 2, order):
    P = inputImage.shape[0]/2
    Q = inputImage.shape[1]/2
    U, V = np.meshgrid(range(inputImage.shape[0]), range(inputImage.shape[1]), sparse=False, indexing='ij')
    Duv = (((U-P)**2+(V-Q)**2)).astype(float)
    #H = 1/(1+(Duv/sigma**2)**order)
    H = 1/1+(sigma/Duv**(2*order))
    return (1 - H)
'''
#.........................Apply homormophic Algorithm...................................
# Frequency Domain (DFT)
def FrequencyDomainTransform(inputImage):
    #inputImage = np.abs(np.fft.fft2(inputImage))
    #inputImage = np.fft.fftshift(inputImage)
    inputImage = np.fft.fft2(inputImage)
    return inputImage

def homormophic_Algorithm(inputImage, filterParam1 = 1, filterParam2 = 1):
    # Apply logarthmic operation and Fourier Transformation on the image
    inputImageLog = np.log1p(np.array(inputImage,dtype="float"))

    inputImageF = FrequencyDomainTransform(inputImageLog)

    # Apply the H.P filters 
    inputImageGaussian = Gaussian(inputImageF,30)
    H = np.fft.fftshift(inputImageGaussian)
    newImage = (float(0.75) + (float(1.25)*H)) * inputImageF
   
    # Apply inverse F.T 
    inputImageGaussianInverse = np.fft.ifft2(newImage)

    
    # Apply Exp. 
    inputImageGaussianInverseExp = np.exp(np.real(inputImageGaussianInverse))-1
  
    return np.uint8(inputImageGaussianInverseExp)


#...............................Test.................................................
#image path 
path = './data/originalImages/test2_original.jpg'

# read image
img = cv2.imread(path)
# Apply Homomorphic algorithm 
imgFiltered = homormophic_Algorithm(img[:,:,0])
cv2.imwrite('./data/homomorpicFilterImages/image_homomorphic2.jpg', imgFiltered)

plt.imshow(img)
plt.show()
plt.imshow(imgFiltered)
plt.show()





