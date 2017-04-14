#**Behavioral Cloning** 

The goal of this project is to predict steering angles for the car and to make it drive autonomously through the simulated track provided by udacity.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_cnn.png "Model Visualization"
[image2]: ./examples/udacity_center_Cam.png 
[image3]: ./examples/folder_5_dataset.png 
[image4]: ./examples/folder_7.png 
[image5]: ./examples/folder_7_augmented.png 
[image6]: ./examples/folder_7_filtered.png
[image7]: ./examples/center_2017_04_06_21_49_48_512.jpg "center lane driving"
[image8]: ./examples/left_2017_04_03_21_20_56_759.jpg "left camera driving"
[image9]: ./examples/right_2017_04_03_21_15_01_158.jpg "right camera driving"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_epochs_trained10.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Model published by nvidia was the starting point for model selection and the model is pictorially represented as 

![alt text][image1]

The model was kept as standard without any changes but the more emphasis was given on the training data and data augmentation to make the car drive autonomously

####2. Attempts to reduce overfitting in the model

The car was driven in the simulator multiple times and this data was used to train the model and the dataset provided from udacity was used to validate the model. The validation and training accuracy was observed to be similar referring the model was not overfitting the data

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

The training started off with just using the center camera data from udacity dataset and splitting it into train and validation sets, the histogram of the dataset for the center camera is as shown below 

![alt text][image2]

Next attempt was by using playstation 4 joystick to record track 1 forward and reverse laps , each 4 in number. THe motivation behind using playstation 4 joystick to simulate near analog performance. The histogram of the data from canter, left and right camera is as shown below

![alt_text][image3]

with just the above dataset the model could easily learn to navigate track 1 after introducing lambda layers to normlize and zero center the pixel of the image. But the thirst of learning wasnt quenched with this. The necessity for generalizing the model to perform for track 2 left a big hole!. Following were the steps taken to make the model generalize the data and are categorised into failed methods and succesful methods 

**Failed attempts**
###Offline Data augmentation 
    For a given dataset, if a particular steering angle has very high biasing effects, Then all the training examples having greater occurances are jittered by translating then only horizontally. For a left shift an angle of 0.005 was added per pixel and 0.005 was subtracted per pixel for right shift
The dataset without augmentation looks like below 

![alt_text][image4]

The dataset after performing augmentation is as shown below 

![alt_text][image5]

The average occurance for each steering angle is calculated. If a particular steering angle has occurances greater than the average, Only average number of occurances are taken and others are dropped.

![alt_text][image6]    
    
The above method introduced a strong right sided steer and model was not predicting steering angles properly.

**Successful attempts**
Inspired by fellow CarND student Vivek Yadav, The method of probabilistically dropping low steering angles was implemented.
If a sample whose steering angle is one the biasing steering angles, then it is jittered until its steering angle is not a biasing steering angle anymore. e.g if steering angle of image_a is zero, image is jittered by horizontally shifting it and adjusting steering angle by 0.005 as discussed before and then given to training.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia model. The motivation behind this was to choose a suitable model and spend more time in understanding and manipulating the data given to the model.

In order to gauge how well the model was working and to check for overfitting problem, 2 tests were done. Once validation set was a just a chunk of training set split using sklearn.train_test_split and other method was to use self recorded data for training and udacity data for validation. In both the cases the model has approximately the same training and validation loss indicating model is not overfitting and augmentation was done correctly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture  consisted of a convolution neural network proposed by Nvidia deep learning paper

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 8 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image7]

The aim was also teach the car to recover from critical situations and not conservatively train it. Left and right cameras along with some recovery manueuvres.

The image from left camera is as shown below  


![alt_text][image8]

A random image from right image is as shown below

![alt_text][image9]

I finally randomly shuffled the data set . Validation set was chosen from udacity provided dataset. Similar training and validation loss was a sign of model is not overfitting the data.

With the above configurations and adjustments the model could succesfully traverse track 1 and 80% of track 2
