# author: Ankith Manjunath
# Date : 02.04.17

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

lines = []
offset_left_right = 0.2
samples = []
data_folder = "../CarND-Behavioral-Cloning-P3/recorded_data/folder_5/"
with open(data_folder + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = data_folder + 'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # #flipped center image
                flipped_image = cv2.flip(center_image, flipCode = 1)
                images.append(flipped_image)
                steering_angle_center_flipped = -1 * center_angle
                angles.append(steering_angle_center_flipped)

                #left camera image
                name = data_folder + 'IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                steering_angle_left = center_angle + offset_left_right
                images.append(left_image)
                angles.append(steering_angle_left)

                # # flipped center image
                flipped_image_left = cv2.flip(left_image, flipCode=1)
                images.append(flipped_image_left)
                steering_angle_left_flipped = -1 * steering_angle_left
                angles.append(steering_angle_left_flipped)

                # right camera image
                name = data_folder + 'IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                steering_angle_right = center_angle - offset_left_right
                images.append(right_image)
                angles.append(steering_angle_right)

                # flipped center image
                flipped_image_right = cv2.flip(right_image, flipCode=1)
                images.append(flipped_image_right)
                steering_angle_right_flipped = -1 * steering_angle_left
                angles.append(steering_angle_right_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
#normalizing the data
from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import ELU
from keras.layers import Dropout

ch, row, col = 3, 160, 320  # camera format

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row,col,ch)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row,col,ch)))
model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))


model.compile(loss = 'mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch = 3 * len(train_samples),
                    validation_data = validation_generator,
                    nb_val_samples = len(validation_samples),
                    nb_epoch = 2,
                    verbose = 1)
model.save('model_comma.h5')


#print(history_object.history.keys())