import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    k_h,k_w = kernel.shape
    output = np.zeros(img.shape)
    # We take p_h = k_h - 1 and p_w = k_w - 1 in order to give the output image the same height and width 
    p_h = (k_h - 1) # We want to pad (k_h - 1)//2 zeros along either side of the rows (y-axis) of the image 
    p_w = (k_w - 1) # We want to pad (k_w - 1)//2 zeros along either side of the columns (x-axis) of the image

    if len(img.shape) > 2: # Case 1 - RGB image with 3 channels
        # We want to pad p = (k_h-1)//2 on both sides of the height of the image, p = (k_w - 1)//2 on both sides along the width of the images and (0,0) specifies that we don't pad along the channels
        padded = np.pad(img, pad_width=((p_h//2, p_h//2), (p_w//2, p_w//2), (0,0)), mode='constant', constant_values=0)
        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                for c in range(img.shape[2]):
                    output[h,w,c] = np.sum(kernel * padded[h:h+k_h, w:w+k_w, c])

    else: # Case 2 - Grayscale image
        padded = np.pad(img, pad_width=((p_h//2, p_h//2), (p_w//2, p_w//2)), mode='constant', constant_values=0) # We don't need to specify (0,0) here since we only have 1 channel
        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                output[h,w] = np.sum(kernel * padded[h:h+k_h, w:w+k_w])

    return output

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel = np.flip(kernel)
    return cross_correlation_2d(img, kernel)

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    gaussian_kernel = np.zeros((height, width))
    for h in range(-height//2, height//2 + 1):
        for w in range(-width//2, width//2 + 1):
            x1 = 1 / (2 * np.pi * (sigma ** 2))
            x2 = np.exp(-(h ** 2 + w ** 2)/(2 * (sigma ** 2)))
            gaussian_kernel[h+height//2, w+width//2] =  x1 * x2
    return gaussian_kernel * (1/np.sum(gaussian_kernel))

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

