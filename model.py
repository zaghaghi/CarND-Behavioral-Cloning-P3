'''
This script read image data and train a model for udacity project
'''
import csv
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D

class Data:
    '''
    This class contains image, steering and other usefull information
    '''
    def __init__(self, image_path, steering, flip=False):
        self.image_path = image_path
        self.steering = steering
        self.flip = flip

class DataGenerator:
    '''
    This class reads directories of recorded data and build a generator
    '''
    steering_correction = 0.2

    def __init__(self, dir_list, valid_split=0.2):
        '''
        contructor
        '''
        self.train_data = []
        self.valid_data = []
        self._read_image_list(dir_list, valid_split)

    def _read_image_list(self, dir_list, valid_split):
        '''
        read image path and steering data from csv files in dir_lists
        '''
        for item in dir_list:
            csv_filename = os.path.join(item, 'driving_log.csv')
            if not os.path.exists(csv_filename):
                continue
            with open(csv_filename) as csv_file:
                reader = csv.reader(csv_file)
                data = []
                for row in reader:
                    # center image and its steering
                    data.append(Data(os.path.join(item, row[0]), float(row[3])))
                    # left image and its steering
                    data.append(Data(os.path.join(item, row[1]), \
                                    float(row[3]) + self.steering_correction))
                    # right image and its steering
                    data.append(Data(os.path.join(item, row[2]), \
                                    float(row[3]) - self.steering_correction))

                    # center flipped image and its steering
                    data.append(Data(os.path.join(item, row[0]), float(row[3]), True))
                    # left flipped image and its steering
                    data.append(Data(os.path.join(item, row[1]), \
                                    float(row[3]) + self.steering_correction, True))
                    # right flipped image and its steering
                    data.append(Data(os.path.join(item, row[2]), \
                                    float(row[3]) - self.steering_correction, True))
        data = shuffle(data)
        self.train_data, self.valid_data =\
            train_test_split(data, test_size=valid_split)

    def _get_data(self, data, batch_size):
        '''
        Generator for generating images
        '''
        num_data = len(data)
        while True:
            data = shuffle(data)
            for offset in range(0, num_data, batch_size):
                batch_data = data[offset:offset+batch_size]
                batch_measurements = []
                batch_images_data = []
                for d in batch_data:
                    img_data = cv2.imread(d.image_path)
                    if d.flip:
                        batch_measurements.append(-d.steering)
                        img_data = np.fliplr(img_data)
                    else:
                        batch_measurements.append(d.steering)
                    batch_images_data.append(img_data)

                yield shuffle(np.array(batch_images_data), np.array(batch_measurements))

    def get_train_data(self, batch_size=32):
        ''' Generator for train data'''
        return self._get_data(self.train_data, batch_size)

    def get_valid_data(self, batch_size=32):
        ''' Generator for validation data'''
        return self._get_data(self.valid_data, batch_size)

    def get_train_size(self):
        '''returns train samples size'''
        return len(self.train_data)

    def get_valid_size(self):
        '''returns validation samples size'''
        return len(self.valid_data)

class Model:
    '''
    This class contains model implementation in keras
    '''
    def __init__(self, data_gen):
        '''Constructor'''
        self.data_gen = data_gen
        self.model = None
        self.build_model()

    def build_model(self):
        ''' Build model '''
        self.model = Sequential()
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
        self.model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Dense(64))
        self.model.add(Dense(1))

    def train_save(self, output, loss='mse', optimizer='adam', nb_epoch=7):
        '''
        Train the model and save it to output filename
        '''
        train_generator = self.data_gen.get_train_data()
        valid_generator = self.data_gen.get_valid_data()

        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.fit_generator(train_generator, \
            steps_per_epoch=self.data_gen.get_train_size(), \
            validation_data=valid_generator, \
            validation_steps=self.data_gen.get_valid_size(), epochs=nb_epoch)
        self.model.save(output)



def main():
    '''
    main function
    '''
    data_dir = [os.path.join('data', d) for d in os.listdir('data') \
                if os.path.isdir(os.path.join('data', d))]
    data_gen = DataGenerator(data_dir)
    model = Model(data_gen)
    model.train_save('model.v2.h5')

if __name__ == '__main__':
    main()
