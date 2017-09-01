**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Apply to video

[//]: # (Image References)
[image0]: ./process_images/calibration_drawlines.jpg "Calibration Drawing"
[image1]: ./cal_result/testundistorted5.jpg "Undistorted Picture Test 5"
[image2]: ./process_images/test5_perspective_transformed.jpg "Road Transformed Test 5"
[image3]: ./process_images/test5_perspective_transformed_sobel.jpg "Road Transformed Test 5 Sobeled"
[image4]: ./process_images/curvature.jpg "Test5 Curvature"
[image5]: ./process_images/histogram.jpg "Test5 Curvature Histogram"
[image6]: ./process_images/curvature_search.jpg "Test5 Curvature Search"
[image7]: ./process_images/overlay.jpg "Test5 Overlay"
[image8]: ./process_images/overlay_with_data.jpg "Test5 Overlay with data"
[video1]: ./project_video.mp4 "Video"

---
advfind.py is the python script with the pipeline for the project. Each section of the rubric is answered under a long strecth of "#" entitled by each goal of the project.

### Camera Calibration

#### 1/2. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

I've used the OpenCV functions: 1. findChessboardCorners and 2. calibrateCamera. A number of images of a chessboard, taken from different angles with the same camera, are the inputs. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. These camera calibration matrix and distortion coefficients can then be used by the OpenCV undistort function to undo the effects of distortion on any image produced by the same camera. Generally, these coefficients will not change for a given camera (and lens). I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like the one above.

#### 4. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectrans()`.   The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. It was apparent that by manipulating HLS and LAB only can already give satisfactory lines detection.

The L channel of the HLS color space is to isolate white lines and the B channel of the LAB colorspace is to isolate yellow lines. To combat the different lighting condition, as apparent with test5 picture, the values of the HLS L channel and the LAB B channel are normalized. Below are examples of thresholds shown after applying color gradients:

![alt text][image3]

#### 5. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions "sliding_window_polyfit" and "polyfit_using_prev_fit" identify lane lines and fit a second order polynomial to both right and left lane lines. The first putes a histogram of the bottom half of the image where the lane lines are and finds the bottom-most x position (or "base"). That is shown below.

![alt text][image5]

Pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second order polynomial to each set of pixels. The image below demonstrates how this process works. The polyfit_using_prev_fit function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. The image below demonstrates this:

![alt text][image6]

#### 6. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated based upon: http://www.intmath.com/applications-differentiation/8-radius-curvature.php. The function named "calc_curv_rad_and_center_dist" does this.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:

```python
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * x_meters_per_pix
```
The car position is the difference between these intercept points and the image midpoint (assuming that the camera is mounted at the center of the vehicle).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

A polygon is generated based on plots of the left and right fits, warped back to the original image using the inverse perspective matrix and overlaid on the image.
This is done by function named "draw_lane". Data for curvature and center position is also overlaid there by function named "draw_data"

![alt text][image7]
![alt text][image8]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Trying to extend the same pipeline to the video took the greatest challenge.

This code likely fail when encountering situation where the lane lines are not on the same pixel colors. In an environment where lighting or color plays a big role such as driving beside a tall white truck, or even shiny surfaced vehicle; snow, covered vehicle, etc. This code is not ready to tackle those situations.

Possible improvement include smart thresholding (dynamically selecting threshold parameters based on certain set of activated pixels).
