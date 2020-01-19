import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Thresholds():
    def __init__(self,img):
        self._img = img
        
    def _abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        #binary_output = np.copy(img) # Remove this line
        grad_binary = sxbinary
        return grad_binary

    def _mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude 
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*gradmag/np.max(gradmag))
        # 5) Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        return mag_binary

    def _dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate gradient direction
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely= np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        absgraddir = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        #binary_output = np.copy(img) # Remove this line
        return dir_binary

    # This function thresholds the S-channel of HLS
    # Use exclusive lower bound (>) and inclusive upper (<=)
    def _hls_select(self,img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        S = hls[:,:,2]
        binary_output = np.zeros_like(S)
        binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        #binary_output = np.copy(img) # placeholder line
        return binary_output
        
    # Edit this function to create your own pipeline.
    def pipeline(self, s_thresh=(170, 255), sx_thresh=(20, 100)):
        # Sobel x
        sxbinary = self._abs_sobel_thresh(self._img, 'x', 3, sx_thresh)  
        # Threshold color channel
        s_binary = self._hls_select(self._img, s_thresh)
        # Stack each channel
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        # Combine thresholds
        combined = np.zeros_like(sxbinary)
        combined[((s_binary == 1) | (sxbinary == 1))] = 1
        #combined[((s_binary == 1) & (sxbinary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined

if __name__ == '__main__':
    test = 5
    image = mpimg.imread('..\\test_images\\test' +str(test)+'.jpg')
    threshold = Thresholds(image)
    result = threshold.pipeline()

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('..\\output_images\\P2output_test' +str(test)+'.png')