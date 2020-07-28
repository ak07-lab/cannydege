import cv2 as cv
import numpy as np
import math

#function to calculate gaussian kernel
def gaussian_kernel(size, sigma):
    size = int(size)//2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0*sigma**2)))/(2.0 * np.pi * sigma ** 2)
    return g

# to extend the image so that first pixel of original image becames the middle pixel for applying gaussian blur
def extend_boundaries(i, j, img, size):
    size = int(size)//2
    up_arr = np.vstack((img[0:size, :], img ))
    upplusdown_arr = np.vstack((up_arr, img[i-size:i, :]))
    left_arr = np.hstack((upplusdown_arr[:, 0:size], upplusdown_arr))
    full_img = np.hstack((left_arr, left_arr[:, j:j+size]))
    return full_img

#gaussian blur to remove noise
def gaussian_blur(g, full_img, size, i, j):
    for k in range(0,i-2):
        for l in range(0, j-2):
            blur_img[k][l] = np.sum(np.multiply(g, full_img[k:k+size, l:l+size]))
    return blur_img

#find gradient magnitude and direction
def gradient(i, j, blur_img):
    global theta, mag
    sobelx = (np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))/8
    sobely = (np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))/8
    up_blur = np.vstack((blur_img[0:1, :], blur_img))
    upplusdown_blur = np.vstack((up_blur, blur_img[i-1:i, :]))
    left_blur = np.hstack((upplusdown_blur[:, 0:1], upplusdown_blur))
    full_img_blur = np.hstack((left_blur, left_blur[:, j:j+1]))
    gx = np.zeros([i,j], np.float32)
    gy = np.zeros([i,j], np.float32)
    theta = np.zeros([i,j], np.float32)
    for k in range(0,i-1):
        for l in range(0,j-1):
            gx[k][l] = np.sum(np.multiply(sobelx, full_img_blur[k:k+3, l:l+3]))
            gy[k][l] = np.sum(np.multiply(sobely, full_img_blur[k:k+3, l:l+3]))
            theta[k][l] = math.degrees(math.atan2(gy[k][l],gx[k][l]))

    mag = np.zeros([i,j], np.float32)
    mag = np.sqrt(np.add(np.square(gx), np.square(gy)))
    mag = mag.astype('uint8')

#suppressing pixel to zero if it is not local maxima in its neighbourhood
def non_maximal_suppresion(i, j, mag, theta):
    theta[theta<0] += 180
    thin = np.zeros_like(img)
    for k in range(0,i-1):
        for l in range(0, j-1):
            try:
                q = 255
                r = 255

                #angle 0
                if  0 <= theta[k][l] < 22.5 or 157.5 <= theta[k][l] <= 180:
                    q = mag[k, l+1]
                    r = mag[k, l-1]
                #angle 45
                elif 22.5 <= theta[k][l] < 67.5:
                    q = mag[k+1, l-1]
                    r = mag[k-1, l+1]
                #angle 90
                elif 67.5 <= theta[k][l] < 112.5:
                    q = mag[k+1, l]
                    r = mag[k-1, l]
                #angle 135
                elif 112.5 <= theta[k][l] < 157.5:
                    q = mag[k-1, l-1]
                    r = mag[k+1, l+1]

                if (mag[k,l] >= q) and (mag[k,l] >= r):
                    thin[k,l] = mag[k,l]
                else:
                    thin[k,l] = 0

            except:
                continue
    return thin

#pixel greater than high threshold are confirmed points of edges and greater than low threshold and lower than high threshold are points than can contribute to edges if its neighbours are strong pixels
def thresholding(i, j, thin):
    final_img = np.zeros_like(img)

    for k in range(0, i-1):
        for l in range(0, j-1):
            if thin[k][l] >= 17:
                final_img[k][l] = 255
            elif 8 <= thin[k][l] < 20:
                if 0 <= theta[k][l] < 22.5 or 157.5 <= theta[k][l] <= 180 and thin[k][l+1] >= 20:
                    final_img[k][l] = 255
                elif 0 <= theta[k][l] < 22.5 or 157.5 <= theta[k][l] <= 180 and thin[k][l+1] >= 20:
                    final_img[k][l] = 255
                elif 22.5 <= theta[k][l] < 67.5 and thin[k+1][l-1] >= 20:
                    final_img[k][l] = 255
                elif 22.5 <= theta[k][l] < 67.5 and thin[k-1][l+1] >= 20:
                    final_img[k][l] = 255
                elif 67.5 <= theta[k][l] < 112.5 and thin[k+1][l] >= 20:
                    final_img[k][l] = 255
                elif 67.5 <= theta[k][l] < 112.5 and thin[k-1][l] >= 20:
                    final_img[k][l] = 255
                elif 112.5 <= theta[k][l] < 157.5 and thin[k-1][l-1] >= 20:
                    final_img[k][l] = 255
                elif 112.5 <= theta[k][l] < 157.5 and thin[k+1][l+1] >= 20:
                    final_img[k][l] = 255
    return final_img


img = cv.imread('lena.jpg', 0)
i = np.size(img, 0)
j = np.size(img, 1)
blur_img = np.zeros([i, j], np.float32)

gradient(i,j,gaussian_blur(gaussian_kernel(5,1), extend_boundaries(i,j,img,5),5,i,j))
edge_img = thresholding(i,j,non_maximal_suppresion(i,j,mag,theta))

cv.imshow('edge_img', edge_img)

cv.waitKey(0)
cv.destroyAllWindows()