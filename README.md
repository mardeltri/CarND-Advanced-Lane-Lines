## **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output_c1.png "Undistorted (calibration)"
[image2]: ./output_images/undistort_test5.png "Distorted correction"
[image3]: ./output_images/undistort_test5_weighted.png "Distorted correction weighted"
[image4]: ./output_images/P2output_test1.png "Transform 1"
[image5]: ./output_images/P2output_test4.png "Transform 4"
[image6]: ./output_images/P2output_test5.png "Transform 5"
[image7]: ./output_images/transform_straight_lines_1.png "Transform straight lines"
[image8]: ./output_images/identify_lines_t3.png "Identify lines"
[image9]: ./output_images/identify_lines_t5.png "Identify lines"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `calibration.py`. This module contains a class called CalibrateCamera whose inputs are: calibration image path and chessboard shape. This class constructor checks if the current camera has been calibrated or not. If it is, it loads the calibration files (mtx, dist); if not, it calibrates the camera.

Camera calibration is carried out by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here it is assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners are detected correctly in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then, the output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  This distortion correction was applied to the test image using the `cv2.undistort()` function and this result was obtained: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Undistortion is carried out using the method undistort defined in the class CalibrateCamera. This one calls this function: 
```python
	cv2.undistort(img, self._mtx, self._dist, None, self._mtx)
```
where img is the image to be corrected and self._mtx and self._dist are computed by cv2.calibrateCamera.

Next picture shows the distorted image and the same undistorted.

![alt text][image2]

Below a weighted image has been depicted in order to check distortion correction.
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)
A class was created in order to compute the thresholded binary image. This class contains different methods and a pipeline which applies those ones. For these project, sobel operator along the x direction and S channel threshold are used. Using these techniques lines are detected quite well as it can be checked below.

![alt text][image4]
![alt text][image5]
![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform consists of a class named PerspectiveTransform with several methods. This code can be found in the file transform.py. Two of those methods are used to wrap or unwrap images.  The `warp()` function takes as inputs an image (`img`). Source (`src`) and destination (`dst`) points were previously defined as class attributes.  I chose the hardcode the source and destination points in the following manner:

```python
    # Four source coordinates
    src = np.float32(
        [[700, 460],
        [1055, 685],
        [254, 685],
        [582, 460]])
    # Four desired coordinates
    dst = np.float32(
        [[1040, 0],
        [1040, 685],
        [250, 685],
        [250, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 700, 460      | 1040, 0       | 
| 1055, 685     | 1040, 685     |
| 254, 685      | 250, 685      |
| 582, 460      | 250, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Line width was set to 1px in order to adjust it precisely.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Having the warped image, the identification of the lane lines is carried out. In order to achieve so, the image is splited horizontally and the histogram considering the points in the vertical direction is computed. Histogram peaks indicate the rectangle center position. This procedure is repeated until the hole image is swept. Thus, the pixels which defines the lanes have been found. The next step consists in fitting a second order polynomial to those points.

The previous procedure is implemented in the file identify_lines.py. Two functions are included:
*fit_polynomial(binary_warped, left_line, right_line): to compute the polynomial coefficients.
*find_lane_pixels(binary_warped): to detect the lane line pixels using the sliding window method.
Two additional functions are found in this file:
*draw_fill(warped,left_line,right_line,_perspective_transformer, real_warped_img): to print the detected lane

![alt text][image8]
![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
