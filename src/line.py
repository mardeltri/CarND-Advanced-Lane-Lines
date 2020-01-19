import numpy as np
from builtins import TypeError
import matplotlib.pyplot as plt
import cv2
from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    MAX_NUM_FITS = 15
    
    PREV_FILTER = 0.95
    NEXT_FILTER = 0.05
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # polynomial coefficients for the last coefficients
        self.all_fits = deque()
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #filtered radius of curvature of the line in some units
        self.filt_radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.x_val_bottom_m = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
    def add_coeffs(self, coeffs):
        if np.any(self.best_fit) != None:
            current_diffs = np.abs(self.best_fit - coeffs)
            if (np.max(current_diffs)>210.0):
                return 
            
        self.current_fit = coeffs
        self.all_fits.append(coeffs)
        num_coeffs = len(self.all_fits)
        
        if (len(self.all_fits)>self.MAX_NUM_FITS):
            self.all_fits.popleft()
            num_coeffs = self.MAX_NUM_FITS
        
        self.best_fit = np.array([0.0,0.0,0.0])
        for coef in self.all_fits:
            self.best_fit += coef
            
        self.best_fit = self.best_fit/num_coeffs
        
    def compute_x_values(self):
        try:
            self.allx = self.current_fit[0]*self.ally**2 + self.current_fit[1]*self.ally + self.current_fit[2]
        except TypeError:
            print('The function failed to fit a line!')
            self.allx = 1*self.ally**2 + 1*self.ally
            
    def draw_lines(self,img):
        pts = np.array([self.allx, self.ally])
        limg = cv2.polylines(img, np.int32([pts.transpose()]), 0, (255,255,0),4)
        return limg
    
    def compute_radii(self):
        fit_cr = np.polyfit(self.ally*self.ym_per_pix, self.allx*self.xm_per_pix, 2)
        y_eval_px = np.max(self.ally)
        y_eval_m = y_eval_px*self.ym_per_pix
        radii = ((1 + (2*fit_cr[0]*y_eval_m + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        self.radius_of_curvature = radii
        if self.filt_radius_of_curvature == None:
            self.filt_radius_of_curvature = radii
        else:
            self.filt_radius_of_curvature = self.PREV_FILTER*self.filt_radius_of_curvature+self.NEXT_FILTER*radii
        print(self.filt_radius_of_curvature)
        # Line positions
        self.x_val_bottom_m = fit_cr[0]*y_eval_m**2 + fit_cr[1]*y_eval_m + fit_cr[2]