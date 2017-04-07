# author: Ankith Manjunath
# Date : 01.04.17

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

lines = []
offset_left_right = 0.25
with open('../CarND-Behavioral-Cloning-P3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:]
images = []
measurements = []
image_folder = '../CarND-Behavioral-Cloning-P3/IMG/'
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = image_folder + filename
    image = cv2.imread(current_path)
    images.append(np.array(image))
    steering_angle_center = float(line[3])
    measurements.append(steering_angle_center)
    # plt.figure()
    # plt.subplot(3,2,1)
    # plt.imshow(np.array(image).squeeze())
    # plt.title(steering_angle_center)

    # flipped_image = cv2.flip(image, flipCode = 1)
    # images.append(flipped_image)
    # steering_angle_center_flipped = -1 * steering_angle_center
    # measurements.append(steering_angle_center_flipped)

    # # plt.subplot(3, 2, 2)
    # # plt.imshow(np.array(flipped_image).squeeze())
    # # plt.title(steering_angle_center_flipped)
    #
    # source_path = line[1]
    # filename = source_path.split('/')[-1]
    # current_path = image_folder + filename
    # left_image = cv2.imread(current_path)
    # #images.append(left_image)
    # steering_angle_left = steering_angle_center + offset_left_right
    #measurements.append(steering_angle_left)
    # plt.subplot(3, 2, 3)
    # plt.imshow(np.array(image).squeeze())
    # plt.title(steering_angle_left)
    #
    # flipped_image = cv2.flip(left_image, flipCode=1)
    # images.append(flipped_image)
    # measurements.append(-1 * steering_angle_left)
    # # plt.subplot(3, 2, 4)
    # # plt.imshow(np.array(flipped_image).squeeze())
    # # plt.title(-1 * steering_angle_left)
    #
    # source_path = line[2]
    # filename = source_path.split('/')[-1]
    # current_path = image_folder + filename
    # right_image = cv2.imread(current_path)
    # images.append(right_image)
    # steering_angle_right = steering_angle_center - offset_left_right
    # measurements.append(steering_angle_right)
    # # # plt.subplot(3, 2, 5)
    # # # plt.imshow(np.array(image).squeeze())
    # # # plt.title(steering_angle_right)
    #
    # flipped_image = cv2.flip(image, flipCode=1)
    # images.append(flipped_image)
    # measurements.append(-1 * steering_angle_right)
    # # plt.subplot(3, 2, 6)
    # # plt.imshow(np.array(flipped_image).squeeze())
    # # plt.title(-1 * steering_angle_right)


X_train = np.array(images)
Y_train = np.array(measurements)
#normalizing the data
from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Flatten(input_shape=(90,320,3)))
model.add(Conv2D(nb_filter = 24, nb_row = 5,nb_col = 5,border_mode ='valid', subsample = (2,2), activation = 'relu'))
model.add(Conv2D(nb_filter = 36, nb_row = 5,nb_col = 5,border_mode ='valid', subsample = (2,2), activation = 'relu'))
model.add(Conv2D(nb_filter = 48, nb_row = 5,nb_col = 5,border_mode ='valid', subsample = (2,2), activation = 'relu'))
model.add(Conv2D(nb_filter = 64, nb_row = 3,nb_col = 3,border_mode ='valid', subsample = (1,1), activation = 'relu'))
model.add(Conv2D(nb_filter = 64, nb_row = 3,nb_col = 3,border_mode ='valid', subsample = (1,1), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer='adam')
history_object = model.fit(X_train,Y_train, validation_split=0.2,shuffle=True, nb_epoch = 2)
model.save('model.h5')


print(history_object.history.keys())