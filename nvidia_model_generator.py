# author: Ankith Manjunath
# Date : 02.04.17

# author: Ankith Manjunath
# Date : 01.04.17

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle

data_folder_train  = "../CarND-Behavioral-Cloning-P3/recorded_data/folder_5/"
data_folder_valid = data_folder_train
#data_folder_valid  = "../CarND-Behavioral-Cloning-P3/recorded_data/udacity/data"
lines              = []
offset_left_right  = 0.2
train_samples      = []
validation_samples = []

with open(data_folder_train + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        train_samples.append(line)

# with open(data_folder_valid + '/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     i = 0
#     for line in reader:
#         if i != 0:
#             validation_samples.append(line)
#         i = 1

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(train_samples, test_size=0.2)

def generator(samples , folder_path , batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = folder_path + '/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                #flipped center image
                # flipped_image = cv2.flip(center_image, flipCode = 1)
                # images.append(flipped_image)
                # steering_angle_center_flipped = -1 * center_angle
                # angles.append(steering_angle_center_flipped)

                #left camera image
                name = folder_path + '/IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                steering_angle_left = center_angle + offset_left_right
                images.append(left_image)
                angles.append(steering_angle_left)

                # flipped center image
                # flipped_image = cv2.flip(left_image, flipCode=1)
                # images.append(flipped_image)
                # steering_angle_left_flipped = -1 * steering_angle_left
                # angles.append(steering_angle_left_flipped)

                # right camera image
                name = folder_path + '/IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                steering_angle_right = center_angle - offset_left_right
                images.append(right_image)
                angles.append(steering_angle_right)

                # flipped center image
                # flipped_image = cv2.flip(right_image, flipCode=1)
                # images.append(flipped_image)
                # steering_angle_right_flipped = -1 * steering_angle_left
                # angles.append(steering_angle_right_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, data_folder_train,  batch_size=32)
validation_generator = generator(validation_samples, data_folder_valid, batch_size=32)
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

nb_epoch = 10
model.fit_generator(train_generator,
                    samples_per_epoch = 6 * len(train_samples),
                    validation_data = validation_generator,
                    nb_val_samples = len(validation_samples),
                    nb_epoch = nb_epoch,
                    verbose = 1)

model.save(data_folder_train + "model_epochs_trained" + str(nb_epoch) + ".h5" )
model.save(data_folder_train + 'model.h5')


#print(history_object.history.keys())