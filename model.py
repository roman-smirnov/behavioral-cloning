import numpy as np
import cv2
import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, GaussianNoise
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2
from keras.layers.pooling import AvgPool2D
from keras import backend as K

IMG_WIDTH = 128
IMG_HEIGHT = 32

L2_PENALTY = 0.001
DROPOUT_PROB = 0.1
NOISE_STDV = 0.1


def load_csv_data():
    headers = ['center', 'left', 'right', 'angle', 'throttle', 'brake', 'speed']
    data = pd.read_csv('./driving_log.csv', names=headers)
    data = data.drop(columns=['left', 'right', 'throttle', 'brake', 'speed'])
    return data


def smooth_data(data):
    ker = np.exp(-np.arange(-12, 13) ** 2 / 8.) / np.sqrt(8. * np.pi)
    ker = ker / np.sum(ker)
    data.angle = np.convolve(data.angle, ker, mode='same')
    return data


def impute_data(data):
    data.angle += np.random.uniform(-0.02, 0.02, len(data))
    data.angle = data.angle.round(decimals=1)
    data.angle.value_counts()
    uniq = data.angle.unique()
    cmax = np.max(data.angle.value_counts())
    for u in uniq:
        rows = data[data.angle == u]
        data = data.append(rows.iloc[np.arange(cmax - len(rows)) % len(rows)])
    return data


def load_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[64:128]
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.Canny(img, 100, 200)
    return img.astype(np.float32)[:, :, None]


def load_x_data(data):
    x_data = np.empty((len(data.center), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    for i in range(len(data)):
        x_data[i] = load_img(data.center.values[i])
    return x_data


def slim_model():
    model = Sequential()
    model.add(Conv2D(4, (5, 5), input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
                     padding='same', activation='relu', kernel_regularizer=l2(L2_PENALTY)))
    model.add(AvgPool2D())
    model.add(GaussianNoise(stddev=NOISE_STDV))
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Conv2D(8, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(L2_PENALTY)))
    model.add(AvgPool2D())
    model.add(GaussianNoise(stddev=NOISE_STDV))
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(L2_PENALTY)))
    model.add(AvgPool2D())
    model.add(GaussianNoise(stddev=NOISE_STDV))
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Conv2D(8, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(L2_PENALTY)))
    model.add(AvgPool2D())
    model.add(GaussianNoise(stddev=NOISE_STDV))
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Conv2D(4, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(L2_PENALTY)))
    model.add(AvgPool2D())
    model.add(GaussianNoise(stddev=NOISE_STDV))
    model.add(Dropout(rate=DROPOUT_PROB))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# prep dataset
data = load_csv_data()
data = smooth_data(data)
data = impute_data(data)
data = shuffle(data)
y_data = data.angle
x_data = load_x_data(data)

# load a model
# model = load_model('./model.h5')

# train the model
model = slim_model()
model.fit(x_data, y_data, batch_size=64, epochs=16, verbose=2, validation_split=0.2)
print('Done Training')

# save the model
model.save('../model.h5', overwrite=True)
print('Model Saved')


