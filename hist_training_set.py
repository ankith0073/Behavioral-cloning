# author: Ankith Manjunath
# Date : 06.04.17

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
data_folder = "../CarND-Behavioral-Cloning-P3/recorded_data/folder_5_extended/"
data_folder_valid  = "../CarND-Behavioral-Cloning-P3/recorded_data/folder_7/"
lines = []
offset_left_right = 0.2
samples = []

steering_angles_test = []
steering_angles_valid = []
with open(data_folder + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        steering_angles_test.append(float(line[3]))
        steering_angles_test.append(float(line[3]) + offset_left_right)
        steering_angles_test.append(float(line[3]) - offset_left_right)

        #steering_angles_test.append(-1 * float(line[3]))
        steering_angles_test.append(-1 * (float(line[3]) + offset_left_right))
        steering_angles_test.append(-1 * (float(line[3]) - offset_left_right))

with open(data_folder_valid + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for line in reader:
        if i != 0:
            samples.append(line)
            steering_angles_valid.append(float(line[3]))
        i = 1

plt.figure()
plt.subplot(2,1,1)
plt.hist(steering_angles_test)
plt.subplot(2,1,2)
plt.hist(steering_angles_valid)
i = 0



