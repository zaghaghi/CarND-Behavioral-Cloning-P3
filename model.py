'''
This script read image data and train a model for udacity project
'''
import csv
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import h5py
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPool2D
from keras.models import load_model
from keras import __version__ as keras_version
import click

class Data:
    '''
    This class contains image, steering and other usefull information
    '''
    def __init__(self, image_path, steering, flip=False):
        self.image_path = image_path.replace('\\', '/')
        self.steering = steering
        self.flip = flip


class DataGenerator:
    '''
    This class reads directories of recorded data and build a generator
    '''
    steering_correction = 0.2

    def __init__(self, dir_list, color_mode=cv2.COLOR_BGR2YUV, valid_split=0.2):
        '''
        contructor
        '''
        self.train_data = []
        self.valid_data = []
        self.color_mode = color_mode
        self._read_image_list(dir_list, valid_split)

    def _read_image_list(self, dir_list, valid_split):
        '''
        read image path and steering data from csv files in dir_lists
        '''
        data = []
        for item in dir_list:
            csv_filename = os.path.join(item, 'driving_log.csv')
            if not os.path.exists(csv_filename):
                print ("path not exists{}".format(csv_filename))
                continue
            with open(csv_filename) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    # center image and its steering
                    data.append(Data(os.path.join(item, row[0].strip()), float(row[3])))
                    # left image and its steering
                    data.append(Data(os.path.join(item, row[1].strip()),
                                     float(row[3]) + self.steering_correction))
                    # right image and its steering
                    data.append(Data(os.path.join(item, row[2].strip()),
                                     float(row[3]) - self.steering_correction))

                    # center flipped image and its steering
                    data.append(Data(os.path.join(item, row[0].strip()), float(row[3]), True))
                    # left flipped image and its steering
                    data.append(Data(os.path.join(item, row[1].strip()),
                                     float(row[3]) + self.steering_correction, True))
                    # right flipped image and its steering
                    data.append(Data(os.path.join(item, row[2].strip()),
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
                    img_data = cv2.cvtColor(img_data, self.color_mode)
                    if img_data is None:
                        print(d.image_path)
                    steering = d.steering
                    if d.flip:
                        steering = -d.steering
                        # img_data = np.fliplr(img_data)
                        img_data = cv2.flip(img_data, 1)

                    batch_measurements.append(steering)
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
    def __init__(self, data_gen, input_model=None):
        '''Constructor'''
        self.data_gen = data_gen
        self.model = None
        self.filename = None
        if not os.path.exists(input_model):
            self.build_model()
        else:
            self.load_model(input_model)

        self._get_next_filename()

    def _get_next_filename(self):
        '''Finds a valid name for model output name'''
        filename = 'model.v{}.h5'
        ver = 0
        while os.path.exists(filename.format(ver)):
            ver += 1
        self.filename = filename.format(ver)
        print("save result model to {}".format(self.filename))

    def build_model(self):
        ''' Build model '''
        self.model = Sequential()
        self.model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                                  input_shape=(160, 320, 3)))
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5))
        self.model.add(Convolution2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Convolution2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Convolution2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1200))
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(1))

    def load_model(self, input_model):
        '''Load a pretrained model'''
        input_model_file = h5py.File(input_model, mode='r')
        model_ver = input_model_file.attrs.get('keras_version')
        keras_ver = str(keras_version).encode('utf8')

        if model_ver != keras_ver:
            print('You are using Keras version ', keras_ver,
                  ', but the model was built using ', model_ver)

        self.model = load_model(input_model)


    def train_save(self, loss='mse', optimizer='adam', epochs=10, batch_size=32):
        '''
        Train the model and save it to output filename
        '''
        train_generator = self.data_gen.get_train_data(batch_size=batch_size)
        valid_generator = self.data_gen.get_valid_data(batch_size=batch_size)

        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.fit_generator(train_generator,
                                 steps_per_epoch=self.data_gen.get_train_size()/batch_size,
                                 validation_data=valid_generator,
                                 validation_steps=self.data_gen.get_valid_size()/batch_size,
                                 epochs=epochs)
        self.model.save(self.filename)


@click.command()
@click.option('--epochs', default=10, help='Number of epochs.', prompt='Number of epochs')
@click.option('--input-dir', default='data', help='Input directory of saved simulations.', prompt='Input directory')
@click.option('--batch-size', default=32, help='Batch size', prompt='Batch size')
@click.option('--input-model', default='', help='Input model', prompt='Input model')
def main(epochs, input_dir, batch_size, input_model):
    '''
    main function
    '''
    data_dir = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, d))]
    data_gen = DataGenerator(data_dir)
    model = Model(data_gen, input_model)
    model.train_save(epochs=epochs, batch_size=batch_size)

if __name__ == '__main__':
    main()
