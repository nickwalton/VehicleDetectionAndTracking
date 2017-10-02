**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./pipeline_images/not_car_example.jpg
[image2]: ./pipeline_images/car_example.jpg
[image3]: ./examples/HOG_example.jpg
[image4]: ./pipeline_images/example_hog.jpg
[image5]: ./pipeline_images/example_spatial.jpg
[image6]: ./pipeline_images/histogram_image.jpg
[image7]: ./pipeline_images/normalization.jpg
[image8]: ./pipeline_images/sliding_window.jpg
[image9]: ./pipeline_images/detected_vehicle.jpg
[image10]: ./pipeline_images/heat_map.jpg
[video1]: ./project_video.mp4

Using computer vision techniques and a sliding window approach to detect vehicles in a video stream. The code for this project is contained in the IPython notebook VehicleTracking.ipynb.

### 1. Training Data

The data I used for this project was provided by Udacity and included images from the GTI and also images extracted from the KITTI data set. The data is loaded in the first cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image2] ![alt text][image1]
 

### 2. Feature Extraction

Although in the past I've used convolutional neural networks to classify traffic signs and in similar situations to get a better understanding of traditional computer vision techniques I used feature extraction techniques such as Histogram Oriented Gradients (HOG) spatial binning and color histograms. My feature extraction functions were all defined in the second cell of my IPython notebook.

The goal with feature extraction is to get meaningful features that can be used by the classifier (in my case a linear SVM) but not so many that it takes too long to train and run your image through the classifier. 

#### Histogram Oriented Gradients

Histogram Oriented Gradients is a method that essentially takes a variable number of pixels in a cell, for example 8x8 and computes the direction of the gradient for each pixel within that cell, or in other words to what side of the pixel is the pixel intensity increasing. For that 8x8 cell a histogram is computed of all these gradients and we the peak of the histogram is taken as the orient direction

I tried out several different combinations of parameters with my classifier and eventuallly decided on the following parameters for my pipeline based on the accuracy I got from my classifier:
`orientations=12`
`pixels_per_cell=(8, 8)` 
`cells_per_block=(2, 2)`

Here is an example using the green channel with these HOG parameters.

<img src='./pipeline_images/hog_example.jpg' width="425"/> <img src='./pipeline_images/car_example.jpg' width="425"/>


#### Spatial Binning and Color Conversion

Using every pixel in an image is probably more information than we need, but there is still value in having pixel intensity information used in our classifier. For my classifier I tried out several different color space conversions and spatial binning parameters. I eventually decided on using the Y channel in the YCrCb color space and resizing the image to a 16x16 image. This kept much of the information about recognizing the feature while still minimizing the number of features it created. Here is an example of the original image and the resize image. 

![alt text][image2] ![alt text][image5]


#### Color Histogram

One possible we could try and find an object in an image is through a technique called template matching. Essentially you have an example image and you try and match that image in the picture. This works if the image looks exactly like the template, but if there are small variations this method fails. However there is a simliar method that is a little more robust and can be useful for us. Instead of trying to match each pixel like in template matching we can take a histogram of the color and that can help us try and match similar objects in the picture. This has some potential downfalls, but it can be a useful feature to give the classiifer more information about the object. 

Here is an example of the output of a color histogram on a car image:

![alt text][image6]

#### Feature Normalization

After all the selected features have been extracted from the image they are all concatenated into one feature vector. However before they are fed into the classifier they need to be normalized. Because they come from several different techniques their scale can be vastly different and without normalization it's possible for one feature to dominate another. Here we can see the feature vector plotted before and after normalization. 

![alt text][image7]

### 3. SVM Linear Classifier

To classify the whether or not a selected window was a car or not I used a linear SVM classifier. A few other possibilities were tested, but a linear SVM was chosen for ease of training and also to help reduce the chance of overfitting possible with higher order dimension SVM's. 

I randomly split the data into 70% training data and 30% test data and was able to achieve about 99% accuracy on the test set giving me confidence that my classifier was working as expected. This is all implemented in the 4th cell of my IPython notebook.

### Vehicle Detection

#### Sliding Window
Now we have a classifier that can take images and decide whether they are cars or not. But there is often multiple cars in an image and we need to know where the car is in the image. To do that we can implement a sliding window approach to look for where the cars are located. To do that I set up several sliding windows that search at different scales and different locations in the image. In general larger windows are used closer to the observing vehicle and smaller windows farther from the it. 

This was one of the most difficult sections for me. I had to spend a signifcant amount of time playing with the size, overlap and number of sliding windows to be able to get somewhat reliable detection. 

Here is an example of just one of the sliding windows with a 50% overlap:


![alt text][image8]

#### Classifying Sliding Window Images

Ultimately I searched on six scales using YCrCb Y channel HOG features plus spatially binned color and histograms of color in the feature vector, to detect the vehicles in the sliding window. I usually got several detections and the image with bounding boxes might look like the following:

![alt text][image9]

#### Merging Detections and Eliminating False Positives

To eliminate false positives and also to merge overlapping detections of the same object I create a heat map with all the detected object bounding box. With the heat map we can set a threshold to elminate false detections and use the centroid of the heat objects to determine where vehicles are:

![alt text][image10]
---

### Video Implementation

Finally I ran my pipeline on a test video to detect the vehicles in it.

Here's a [link to my video result](./output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the weakest point in my project was the sliding window approach. Often a vehicle wasn't well detected or the bounding box was wobbly due to the placement of the sliding windows. And the more I increased the number of sliding windows the computation time greatly increased so simply increasing the sliding windows wasn't a good approach. What I could have done however to reduce that would be to implement a HOG across the whole image and then only draw upon that for features for each window. In fact that could probably have been done for spatial binning as well.

The other thing I could have done is add more tracking algorithms so that the object is tracked across multiple frames. This could have helped the tracker from losing the car for some frames, reduced false detections further and also help the boxes stay more stable as the car moved across the scene. 

