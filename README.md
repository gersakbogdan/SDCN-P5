#**Self-Driving Car Engineer Nanodegree**

##**Vehicle Detection and Tracking Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/car_notcar.png "Car & Not-Car"
[image2]: ./output_images/ch1.png "CH-1"
[image3]: ./output_images/ch2.png "CH-2"
[image4]: ./output_images/ch3.png "CH-3"

[image5]: ./output_images/test1_2_vehicle_detection.png "Test Image Vehicle Detection"
[image6]: ./output_images/test3_4_vehicle_detection.png "Test Image Vehicle Detection"
[image7]: ./output_images/test5_6_vehicle_detection.png "Test Image Vehicle Detection"

[image8]: ./output_images/test1_heatmap.png "Test Image 1 Heat Map"
[image9]: ./output_images/test2_heatmap.png "Test Image 2 Heat Map"
[image10]: ./output_images/test3_heatmap.png "Test Image 3 Heat Map"
[image11]: ./output_images/test4_heatmap.png "Test Image 4 Heat Map"
[image12]: ./output_images/test5_heatmap.png "Test Image 5 Heat Map"
[image13]: ./output_images/test6_heatmap.png "Test Image 6 Heat Map"

[image14]: ./output_images/test_combined_heatmap.png "Test Combined Heat Map"

[video1]: ./output_images/project_video_result.gif "Video Result"

---
###Files

My project includes the following files:
* vehicle_detection.py - contains all necessary functions to detect vehicles
* vehicle_detection.ipynb - interactive notebook to apply and vizualize vehicle detection in images and videos
* writeup_report.md summarizing the results

---
###Histogram of Oriented Gradients (HOG)

The code for this step is contained in lines `16` through `33` of the file called `vehicle_detection.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]
![alt text][image4]

I tried various combinations of parameters and color channels such as RGB, HSV, LUV, HLS, YUV, YCrCb.

However for extracting the HOG features I decided in the end to use all YCrCB transformation, 3 channels, 9 orientations, 8 pixels per cell and 2 cell per blocks, (32, 32) spatial bins and 32 histogram bins.

The main reason for this was because I found this settings working pretty well on both test images and video.

```
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size =  (32, 32)
hist_bins = 32
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
```

####SVM Classifier

Next step is to train a classifier to detect cars and not cars images (`train_classifier` function from `vehicle_detection.py` file).
For this I used `sklearn.svm` to train a SVM classifier using `LinearSVC` function.
The data set used for training contains labeled images with two categories, cars and not cars.
After extracting HOG features and normalize them using `StandardScaler` we end up with a feature vector with length 840 for each image in our data set.

Here is the output for applying SVM classfier on our training set:

```
113.96 Seconds to extract HOG features...
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
50.29 Seconds to train SVC...
Test Accuracy of SVC =  0.9879
```

---
###Sliding Window Search

For this step I used the method presented in `Hog Sub-sampling Window Search` lesson which allows us to only extract hog features once and then to sub-sampled to get all of its overlaying windows (code is available under `find_cars` function from `vehicle_detection.py` file).
Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance.
The final parameters used are:
```
ystart = image.shape[0] / 2
ystop = image.shape[0]
scale = 1.5
pix_per_cell = 8
cell_per_block = 2
cells_per_step = 2
```

Ultimately appling the above pipeline using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector I got the following results on test images:

![alt text][image5]
![alt text][image6]
![alt text][image7]

---
###Thresholding and False Positives

The next step is to remove the false positive and combine the overlaping detection.
In order to do this I recorder the positions of positive detection in each frame and  I created a heat map (`add_heat` function from `vehicle_detection.py` file) and apply a threshold (`apply_threshold` function from `vehicle_detection.py` file) to it to allow output where there has been an overap of at least 2.
I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap applied to the test images:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

Applying only on one frame at a time is good but not really helpfull the best results are optained when we apply this technique on multiple consecutive frames.
For example, this is the result optained after applying this technique on the integrated heatmap from the last 6 frames:

![alt text][image14]

---
###Pipeline (video)

And this is the final output using the above pipeline applied on `project_video.mp4` file:

![alt text][video1]

Here's a [link to my video result](./output_videos/project_video_result.mp4) (mp4).

---
###Discussion

This was very challenging project but now I have a better understanding about how to detect objects, in our case cars, inside an image or series of images (video).

My approch was to apply all the techniques discussed in the lessons, with small adjustments, and try to figure it out which one works best in my case.

The final pipeline is doing a fair job on the project video but there are a lot to improve.

There are places where the car is not detected and also at some point the pipeline fails to detect multiple cars when there is some overlapping.

I would also like to try to implement this using deep learning classifier to be able to compare the results.
