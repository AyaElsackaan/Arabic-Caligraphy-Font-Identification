from skimage.feature import greycomatrix, greycoprops
from skimage.filters import gabor_kernel
import numpy as np
import math
from scipy import ndimage as nd
import cv2
from skimage import data, color, feature,morphology
from pre_processing import *
from commonfunctions import *
from collections import Counter
from scipy.signal import argrelextrema
from scipy.signal import convolve2d




def GLCM_features(gray):
    gray = gray.astype(int)
    gmatr = greycomatrix(gray, [1], [0,math.pi/4,math.pi/2,(3/4)*math.pi], levels = 2, normed = True)
    contrast = greycoprops(gmatr, 'contrast')
    correlation = greycoprops(gmatr, 'correlation')
    energy = greycoprops(gmatr, 'energy')
    homogeneity = greycoprops(gmatr, 'homogeneity')
    return [contrast,correlation, energy, homogeneity]

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        filtered=filtered/np.max(filtered)
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats

def gabor_filter(img):
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for frequency in (4,8,16,32):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=1, sigma_y=1))
            kernels.append(kernel)
    return compute_feats(img, kernels)

def get_max_theta(img):
    img = morphology.skeletonize(crop_image(pre_process(img)))
    #print(img.shape)
    #show_images([img])
    lines = cv2.HoughLines(img, 1, np.pi / 180, 20)
    #print(lines[:,:,1].reshape(-1))
    b = Counter(lines[:,:,1].reshape(-1))
    return [a for a,b in b.most_common(20)]

def skew_detection(img):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(img)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    return angle

def project_image_diagonal(img):
    a=crop_image(pre_process(img)).astype('uint8')
    #show_images([a])
    return cv2.resize(np.array([np.trace(a, offset=i) for i in range(-np.shape(a)[0] + 1, np.shape(a)[1])]).astype("uint8"), (50,1), interpolation = cv2.INTER_AREA)


def HVSL(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
    #print(lines)
    if (lines is None):
        return -1
    HV = 0;
    Total = 0
    for line in lines:
        if (line[0, 1] == 0 or abs(line[0, 1] - np.pi / 2) < 0.001):
            HV += 1;
        Total += 1

    return HV / Total


def HPP(binary_image):
    # de beta5od binary image w betraga3 histogram bey3ed fe kol line fe kam pizel 1 w da bey3abar 3an 2el base line
    x = np.sum(binary_image, axis=1)
    x = x / np.max(x)
    return x


def get_max_vertical(binary_image):
    # de beta5od binary image w betraga3 histogram bey3ed fe kol line fe kam pizel 1 w da bey3abar 3an 2el base line
    np.sum(binary_image, axis=0)
    return np.sum(binary_image, axis=0) / binary_image.shape[0]


def HPP_Skeletonize(binary_image):
    # de beta5od binary image w betraga3 histogram bey3ed fe kol line fe kam pizel 1 w da bey3abar 3an 2el base line
    x = np.sum(morphology.skeletonize(binary_image), axis=1)
    x = x / np.max(x)
    return len(argrelextrema(x, np.greater)[0]) / binary_image.shape[0]
    # print(np.array(list(argrelextrema(x, np.greater))).shape[1])
    # maximas=np.sort(x[argrelextrema(x, np.greater)[0]])
    # base_lines=[maximas[-1]]
    # return len(argrelextrema(x, np.greater)[0])
    #     if(len(maximas)>1):
    #         base_lines.append(maximas[-2])
    #     else:
    #         base_lines.append(0)
    #     if(len(maximas)>2):
    #         base_lines.append(maximas[-3])
    #     else:
    #         base_lines.append(0)
    #     if(len(maximas)>3):
    #         base_lines.append(maximas[-4])
    #     else:
    #         base_lines.append(0)
    #     if(len(maximas)>4):
    #         base_lines.append(maximas[-5])
    #     else:
    #         base_lines.append(0)
    #     if(len(maximas)>5):
    #         base_lines.append(maximas[-6])
    #     else:
    #         base_lines.append(0)
    return base_lines


def get_histogram_of_gradients(img):
    img = np.float32(img) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    _, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = np.histogram(angle)[0] / (angle.shape[0] * angle.shape[1])
    #print(hist)
    return hist / hist.max()


def center_of_mass(img, scale=1):
    partition_size_x = img.shape[0] // scale
    partition_size_y = img.shape[1] // scale
    # centers_x=[]
    # centers_y=[]
    # for row in range(1,img.shape[0],partition_size_x):
    #    for col in range(1,img.shape[1],partition_size_y):
    # parition=img[row:row+partition_size_x,col:col+partition_size_y]

    mass_x, mass_y = np.where(img == 1)
    # mass_x and mass_y are the list of x indices and y indices of mass pixels

    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    #             if cent_x is None:
    #                 print(parition)
    #             centers_y.append(cent_y)
    #             centers_x.append(cent_y)
    return cent_x / img.shape[0]


def get_corners(gray):
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # print(dst)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    img = gray.copy()
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.3 * dst.max()] = 255
    # plt.imshow(img)
    #print(np.round(np.sum(dst > 0.3 * dst.max()), 5) / np.round(np.sum(dst > 0.01 * dst.max()), 5))
    return np.round(np.sum(dst > 0.3 * dst.max()), 5) / np.round(np.sum(dst > 0.01 * dst.max()), 5)


def sift_discriptor(img):
    sift = cv2.SIFT_create()
    image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    keypoints, descriptors = sift.detectAndCompute(image8bit, None)
    sift_image = cv2.drawKeypoints(image8bit, keypoints, img)
    #print(descriptors.shape)
    plt.imshow(sift_image)
    return descriptors.shape[0] / (img.shape[0] * img.shape[1])


def count_ones(img):
    return np.sum(img) / (img.shape[0] * img.shape[1])


def LBP(image, eps=1e-7, numPoints=10, radius=8, window=100, method="uniform"):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    histograms = []
    # image = rgb2gray(image)
    # image = cv2.Canny(np.uint8(image),50,150,apertureSize = 3)
    image = image.reshape(image.shape[0], image.shape[1])
    for i in range(1, image.shape[0], window):
        for j in range(1, image.shape[1], window):
            partition = image[i:i + window, j:j + window]
            lbp = feature.local_binary_pattern(partition, numPoints, radius, method)
            # print(lbp.shape)
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, numPoints + 3),
                                     range=(0, numPoints + 2))
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            histograms.append(hist)
    # return the histogram of Local Binary Patterns
    histograms = np.array(histograms).flatten()
    # print(hist)
    #print(histograms)
    return histograms


# LVL used to separete Kufi and Square Kufi
def LVL(gray):
    verticalLines = []  # stores the theta and prependicular of the vertical lines
    Vertical = 0  # number of vertical lines
    Total = 0  # number of  lines
    verticalLinHeights = []  # will contain the lengtn of vertical lines
    maxY = -math.inf  # will have the max y value
    minY = math.inf  # will have the min y value

    #     #reshaping the image to 3D
    # #     gray=gray.reshape(gray.shape[0],gray.shape[1],1)
    # #     print(gray.shape)
    #     show_images([gray])

    # turning the black to white writing & cropping text only
    img = pre_process(gray)
    img = crop_image(img, tol=0)

    # reshaping into 2D
    img = img.reshape(img.shape[0], img.shape[1])

    # converting to skeleton
    skelImg = morphology.skeletonize(img, method='zhang')
    skelImg = skelImg.astype('uint8')

    # getting all lines in the image
    lines = cv2.HoughLines(skelImg, 1, np.pi / 180, 80)
    linesLength = cv2.HoughLinesP(skelImg, 1, np.pi / 180, 5, minLineLength=10, maxLineGap=10)
    # print("this is the lines",lines)

    # if no lines is found
    if (lines is None):
        return [-1]

    # counting the vertical lines
    for line in lines:
        if (abs(line[0, 1] - np.pi / 2) < 0.001):
            Vertical += 1;
            verticalLines.append(line)
        Total += 1

    # gething the length of vertical lines
    #     if (linesLength is None):
    #         return -1

    for line in linesLength:
        x1, y1, x2, y2 = line[0]
        # geting the highest Y and lowest Y to detemine text height
        maxY = max(y1, y2, maxY)
        minY = min(y1, y2, minY)

        # getting vertical lines length
        if (x1 - x2 == 0):
            verticalLinHeights.append(abs(y1 - y2))

    # (a) the text height from the bottomto top
    TextHeight = abs(maxY - minY)

    # (c) the length of the highest detectedvertical line
    maxLength = max(verticalLinHeights)

    # (d) the difference ratio between the text height and the highest vertical line
    ratio_TextHeight_MaxLine = maxLength / TextHeight

    # (e) the variance among the vertical lines
    norm_verticalLinHeights = verticalLinHeights / TextHeight

    return [ratio_TextHeight_MaxLine]



def lpq(img,winSize=3,freqestim=1,mode='nh'):
    rho=0.90

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA=8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
        w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode=='im':
        LPQdesc=np.uint8(LPQdesc)

    ## Histogram if needed
    if mode=='nh' or mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode=='nh':
        LPQdesc=LPQdesc/LPQdesc.sum()

    #print(LPQdesc)
    return LPQdesc
def hu_moments_func(test_image):
    return cv2.HuMoments(cv2.moments(test_image)).flatten()
