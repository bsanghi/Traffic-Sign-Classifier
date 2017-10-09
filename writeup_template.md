#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is ?
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

The distribution of classes are very skewed for all data sets. You can find plots in jupyter notebook.
The augmentation is needed for some classids which is ordered in panda table.
Second, 32x32 resolution is not good enough for some classids. expect lower probability for some classes.
You can see it from example sign plots. 

###Design and Test a Model Architecture

####1.1 Augmentation

The augmentation is needed for classes with smaller datasets. However, the test accuracy is almost 96% without augmentation when I used the deeper model(32 and 64 conv layers) instead of the original lenet model.

For the analysis, i used augmented data(4 x training data)  and saw little bit accuracy improvement. 

####1.2 Color scale and Normalization
I tried several diffirent normalizations. The performence with any normalization is much better than the performence without
normalization.  The following normalization gave me the best result. But, their results are quite close.
Also, the gray scale gave better accuracy and learnt faster. 

####2. I modified the lenet model and increased the number of conv layers. The idea was mentioned in papers and blogs (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). 
Also, i got some ideas and codes from blogs and github accounts.
https://navoshta.com/traffic-signs-classification/
https://github.com/navoshta/traffic-signs/blob/master/Traffic_Signs_Recognition.ipynb
https://github.com/tdanyluk/CarND-Traffic-Sign-Classifier-Project

I did study tuning parameters below.

conv1_num,conv2_num, fill1_input,fill2_input,fill3_input,    traing, valid, test accuracy
16,       32,         800,       120,         80         --- 0.973,0.970,0.942
32,       64          1600,      120,         80         --- 0.988,0.979,0.956
64,       128,        3200       120          80         --- 0.994,0.988,0.957

You can see 16 and 32 conv layers can give significant increase. I chose 32 and 64 for the project.

The most accuracy improvement is due to deepening the model. I tuned the most of parameters.

dropout parameter:

keep_prob = 0.5 to reduce overfitting. when i use larger number, difference between test and validation accuracies increased. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image      							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Flatten       		| Input = 5x5x64. Output = 1600					|
| dropout   			| 0.5 											|
| Fully connected		| Input = 1600. Output = 240					|
| RELU					|												|
| dropout   			| 0.5 											|
| Fully connected		| Input = 240. Output = 90  					|
| RELU					|												|
| dropout   			| 0.5 											|
| Fully connected		| Input = 90 Output = 43  					|
| RELU					|												|
| dropout   			| 0.5 											|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer and the following parameters:

rate = 0.001
EPOCHS = 30
BATCH_SIZE = 128
sigma = 0.1

####4. Having deeper model really helps. Human-level labeling can be worse than the result.  

My final model results were:

Training Accuracy = 0.999
Validation Accuracy = 0.991
Test Accuracy = 0.975

I started from modifying lenet and tried to make it deeper(32 and 64 conv layers) after reading the previous blogs and github projects 
for carnd. Dropout is added for regularization. keep_prob=0.5. I tuned all parameters. the above parameter combination gave me the best result.


###Test a Model on New Images

####1. I chose 8 example signs found from the web  and divided to two directories, test_images and new_images. 
The images for test_images were to used to check if our predictor is working. The images for new_images are used 
to explain shortcomings of our predictor.

1.2 Lets look at images in the images in test_images. The images have clean random only one sign which is centered.
These images should be identified clearly or atleast should be found in top choices.

| Image                         |     Prediction                                                        | 
|:---------------------:|:---------------------------------------------                                 | 
| Road work                     | Road work                                                             | 
| Yield                         | Yield                                                                 |
| Bewar of ice/snow             | Bewar of ice/snow                                                     |
| Slippery road                 | Wild animals crossing                                                 |
| Speed limit (30km/h)          | Speed limit (30km/h)                                                  |

Predicted: Road work (CORRECT)
1.0000 Road work
0.0000 Bumpy road
0.0000 Beware of ice/snow
0.0000 Bicycles crossing
0.0000 Priority road

Predicted: Yield (CORRECT)
1.0000 Yield
0.0000 Keep right
0.0000 Priority road
0.0000 Turn left ahead
0.0000 Go straight or right

Predicted: Beware of ice/snow (CORRECT)
0.9989 Beware of ice/snow
0.0011 Right-of-way at the next intersection
0.0000 Keep right
0.0000 Slippery road
0.0000 Priority road

Predicted: Wild animals crossing (INCORRECT, expected: Slippery road)
0.6504 Wild animals crossing
0.3456 Slippery road
0.0025 Bicycles crossing
0.0010 Road work
0.0004 Bumpy road

Predicted: Speed limit (30km/h) (CORRECT)
1.0000 Speed limit (30km/h)
0.0000 Speed limit (100km/h)
0.0000 Speed limit (50km/h)
0.0000 Speed limit (20km/h)
0.0000 Speed limit (70km/h)


In the top, top 5 choices and corresponding probablities are shown. Only one image(slippery sloep) is misidentified.
Since statistical uncertainty for this sample(only 5 images) is large, it will be always consistent with our results(97.5%).

test_accuracy = 0.975             new_test_sample=0.80        stat_err~sqrt(5)/5

If we use more data(thousands images), the above method works better and shows clearly if our method is working. In this case, 
using probability of each image makes much more sense. Lets calculate probability of misidentifing one picture. 

If you look at top 5 probabilities, all correctly predicted images have close to 1 probabilities.
But, the misidentified image(slippery slope) have :

Predicted: Wild animals crossing (INCORRECT, expected: Slippery road)
0.6504 Wild animals crossing
0.3456 Slippery road

So, probability of this kind of combinations we can see is 1*1*1*1*0.340=0.34 which is not odd case. This kind of combinations
happens little more than (1/3) when we use images with similar qualities. 

So, I conclude that our predictor is working well if sample images fits certain requirements(only one sign, centered).

2. Lets look at complicated cases which does not fit the above requerment. The images are found in new_images.
Lets look at image by images
1) The miscentered speed limit(50) : Its clearly misidentified. Our predictor is not working when sign is miscentered.

Predicted: Children crossing (INCORRECT, expected: Speed limit (50km/h))
0.4849 Children crossing
0.1078 Pedestrians
0.0500 Speed limit (30km/h)
0.0420 Beware of ice/snow
0.0367 Road narrows on the right

2) The completely new sign which is not in our training dataset. Ofcourse, it should misidentify. 

Predicted: Turn left ahead
0.6184 Turn left ahead
0.1526 Bumpy road
0.0974 Bicycles crossing
0.0797 Road work
0.0187 No passing

3) The two sign plates(example8.jpg). Its centered. But, the smaller sign(1km ahead) located below the main sign(slippery road)
The sloppery sign is not found in the top 5. But, speed limit 100 and 120km/h are in top 5 because they have the same digit(1)
found in the smaller sign. Our predictor pick up numbers quite well.

Predicted: Speed limit (100km/h) (INCORRECT, expected: Slippery road)
0.5531 Speed limit (100km/h)
0.2730 Priority road
0.1617 Roundabout mandatory
0.0055 Stop
0.0041 Speed limit (120km/h)

So, I conclude that our predictor is not working well for miscentered images and images with more than 1 sign.
Second, whenever image is misidentified, all top choices have significant probablities. In other case, probability for top 1 has 
close to 1.0 and probabilitis for other choices are close to 0.0 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


