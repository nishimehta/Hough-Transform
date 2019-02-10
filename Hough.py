#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:22:42 2018

@author: nishimehta
#person 50291671
#ubit nishimeh
"""

import cv2
import numpy as np

# performs convolution between matrix m and k
def convolution(m,k):
    product=0
    n=len(m)-1
    for i in range(len(m)):
        for j in range(len(m)):
            product+=(m[i][j] * k[(n-i)][n-j])
    return product

# extracts matrix of size nxn from matrix mat around position i,j
def extract_matrix(i,j,mat,n):
    x = int(n/2)
    extract=[[0 for x in range(n)] for y in range(n)] 
    for k in range(n):
        for l in range(n):
            extract[k][l] = mat[i-x+k][j-x+l]
    return extract

#applies the filter to the image
def apply_mask(im,mask):
    x = int((len(mask)/2))
    convolved = np.zeros(im.shape)
    # ignores boundary values
    for i in range(x,len(im)-x):
        for j in range(x,len(im[0])-x):
            # extracts matrix for each pixel
            m = extract_matrix(i,j,im,len(mask))
            # result is the convolution
            convolved[i][j] = abs(convolution(m,mask))
    return convolved

def erosion(kernel,im):
    eroded = np.zeros(im.shape)
    s = int(len(kernel)/2)
    for i in range(s,len(im)-s):
        for j in range(s,len(im[0])-s):
            p = extract_matrix(i,j,im,len(kernel))
            eroded[i][j] = 255 if np.array_equal(p,kernel) else 0
    return eroded

def dilation(kernel,im):
    dilated = np.zeros(im.shape)
    s = int(len(kernel)/2)
    for i in range(s,len(im)-s):
        for j in range(s,len(im[0])-s):
            p = extract_matrix(i,j,im,len(kernel))
            dilated[i][j] = 255 if (p == kernel).any() else 0
    return dilated

def hough_line(img,u=90,l=-90):
  # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(l,u,2))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
          # Calculate rho. diag_len is added for a positive index
          rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
          accumulator[rho, t_idx] += 1

    #for peaks in the accumulator
    kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    accumulator = apply_mask(accumulator,kernel)
    return accumulator, thetas, rhos


def hough_circle(img,r=22,u=360,l=0):
    #using a fixed radius approach and calculating a and b values
    # Theta ranges
    thetas = np.deg2rad(np.arange(l,u))
    width, height = img.shape

    # Cache some resuable values
    cos_t = r*np.cos(thetas)
    sin_t = r*np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of a vs b
    accumulator = np.zeros((height+r, width+r))
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
          a = int(round(x - cos_t[t_idx]))
          b = int(round(y - sin_t[t_idx]))
          
          accumulator[a,b] += 1

    #for peaks in the accumulator
    kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    accumulator = apply_mask(accumulator,kernel)
    return accumulator


# sobel filter around x axis
sobel_x=[[1,0,-1],[2,0,-2],[1,0,-1]]
#sobel filter around y axis
sobel_y=[[1,2,1],[0,0,0],[-1,-2,-1]]
# reads image to perform operation in gray scale
image = cv2.imread('hough.jpg',0)
original = cv2.imread('hough.jpg')

# detect edges with sobel x and saves the output image
gradient_x=apply_mask(image,sobel_x)
cv2.imwrite('sobel.jpg',gradient_x)

# thresholding and making binary image
gradient_x[gradient_x>70] = 255
gradient_x[gradient_x<71] = 0
cv2.imwrite('filtered.jpg',gradient_x)

# applying morphological operations to the image to remove noise
strut = 255*np.ones((3,3))
dilated = dilation(strut,gradient_x)
eroded = erosion(strut,dilated)
eroded = erosion(strut,eroded)
eroded = dilation(strut,eroded)
binary = erosion(strut,eroded)
cv2.imwrite('dilated.jpg',binary)

# restricting theta values to obtain vertical lines only
upper = 10
lower = -10

accumulator, thetas, rhos = hough_line(binary,upper,lower)
# taking highest 15 values from the accumulator
idx,idy = np.unravel_index(np.argsort(accumulator, axis=None)[-15:][::-1], accumulator.shape)
toprhos = rhos[idx]
topthetas = thetas[idy]

#eliminating similar values 
toprhos_r = np.around(toprhos, decimals=-1)
x, indices = np.unique(toprhos_r, return_index=True)
toprhos = toprhos[indices]
topthetas = topthetas[indices]
print('Number of red lines detected:',len(toprhos))
print('Rho:',toprhos,'Theta:',np.rad2deg(topthetas))

#plotting lines
for rho,theta in zip(toprhos,topthetas):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(original,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('red_line.jpg',original)

# restricting theta values to obtain diagonal lines only
upper = -30
lower = -50

accumulator, thetas, rhos = hough_line(binary,upper,lower)
# taking highest 20 values from the accumulator
idx,idy = np.unravel_index(np.argsort(accumulator, axis=None)[-20:][::-1], accumulator.shape)

toprhos = rhos[idx]
topthetas = thetas[idy]

#eliminating similar values 
toprhos_r = np.around(toprhos, decimals=-1)
x, indices = np.unique(toprhos_r, return_index=True)
toprhos = toprhos[indices]
topthetas = topthetas[indices]

print('Number of blue lines detected:',len(toprhos))
print('Rho:',toprhos,'Theta:',np.rad2deg(topthetas))
#plotting lines
original = cv2.imread('hough.jpg')
for rho,theta in zip(toprhos,topthetas):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(original,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('blue_lines.jpg',original)

#binarizing the image using thresholding
original = cv2.imread('hough.jpg')
gradient_x=apply_mask(image,sobel_x)
gradient_x[gradient_x>100] = 255
gradient_x[gradient_x<100] = 0

# detecting coins
circles = hough_circle(gradient_x)

# taking highest 38 values from the accumulator
maxcirc = 38
centers = np.asarray(np.unravel_index(np.argsort(circles, axis=None)[-maxcirc:][::-1], circles.shape))

#eliminating similar values in the accumulator
centers = np.reshape(centers.ravel(), (maxcirc,2), order='F')
centers_r = np.around(centers, decimals=-1)
centers_r,i = np.unique(centers_r,return_index=True,axis=0)
centers = centers[i]

centers = tuple(map(tuple, centers))
#plotting circles of fixed radius
r = 22
for center in centers:
  cv2.circle(original,center,r,(0,0,255),2)
cv2.imwrite('coin.jpg',original)


