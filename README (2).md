# Final Project of 31392 - Perception of Autonomous Systems
## Project Goals
- Calibrate and rectify the stereo input.
- Process the input images to detect objects on the conveyor and track them in 3D, even under occlusion.
- Train a machine learning system that can classify unseen images into 3 classes (cups, books and boxes) based either on 2D or 3D data.

## ToDo
- [x] 4/9/2020 Calibrate and Rectify the stereo input
- [x] 4/23/2020 Object Tracking
- [ ] 4/24/2020-5/1/2020 Classification
- [ ] 5/2/2020-5/10/2020 Report

## Prerequisite
- python == 3.7.4
- opencv == 3.4.2
- imutils == 0.5.3

## Part1: Calibration and Rectification
### opencv function
- calibrateCamera()
- undistort()
- stereoCalibrate()
- initUndistortRectifyMap()
- remap()
- stereoRectify()

### process
1. Calibrate the camera
2. Undistort the checkboard images
3. Recalibrate the camera
4. Undistort the conveyor images
5. Rectification

## Part2: Track Object
### opencv function
- calcOpticalFlowPyrLK()
- calcOpticalFlowFarneback()
- dilate()
- erode()
- findContours()
- threshold()
- createBackgroundSubtractorKNN()
- contourArea()
- boundingRect()

### assumptions
- Assumption1: As we do not consider a sub-pixel movement here. If an object moves in a speed less than 1 pixel in a sample time, it should be considered static. 
- Assumption2: The movement of the object can be seen as continuous spatially. 
- Assumption3: Since the object moves on a conveyor, its path is approximate a line.  

### ideas
- I have two ideas based on the dense optical flow
- TemplateDenseOpticalFlow is to put a kernel run on the whole image(like CNN), which can score the regions it covers. The score is based on dense optical flow. Of course without the backend optimization it is very time-consuming. Of course it should not be used to track the object, since the result is not real-time. But with a proper kernel and stride, it shall locate the object(get a ROI) accurately. Currently, I'm not planning to add scripts to recognize 2 objects in a frame. 
- OpticalPyramid is inspired by the Gaussian pyramid in SIFT, etc. The pyramid is consist of layers that locate the moving objects in an image. The dilation kernel size and ROI changes between the layers. In each layer, the model shall select a ROI in the image and crop it. As the level of the pyramid increases, the image will become smaller and the dilation kernel size will be lower, too. A smaller dilation kernel is often linked with a larger intensity of the "flow figure" got from dense optical flow function. That is, more movements will be visualable.

## Part3: Classification
### images from Internet
The most convenient way to get a number of images is to use a crawer. See fetchImages.py for more details. You may need
to provide a keyword, e.g., 'red book', and the number of images. 

### now we have datasets...
I plan to use a resNet to classify the images. 

### more
- A pretained model from the internet?
- Data argumentation?
- Classic ML model such as SVM?
 










