
import os
import sys
import h5py
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join('.', '..')))

from keras.models import Sequential
from keras.layers import (
    Flatten,
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D
)
from keras import backend as K

K.set_image_data_format('channels_last')


def VGG_16(weights_path=None, shape=(48, 48)):

    # ========================================================
    # CPU-FRIENDLY MODEL
    # TensorFlow uses NHWC:
    # (batch, height, width, channels)
    # ========================================================

    model = Sequential()

    model.add(
        ZeroPadding2D(
            padding=(1, 1),
            input_shape=(48, 48, 1)
        )
    )

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            name='conv1'
        )
    )

    model.add(
        ZeroPadding2D(
            padding=(1, 1)
        )
    )

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            name='conv2'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(
        ZeroPadding2D(
            padding=(1, 1)
        )
    )

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation='relu',
            name='conv3'
        )
    )

    model.add(
        ZeroPadding2D(
            padding=(1, 1)
        )
    )

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation='relu',
            name='conv4'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(
        ZeroPadding2D(
            padding=(1, 1)
        )
    )

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation='relu',
            name='conv5'
        )
    )

    model.add(
        ZeroPadding2D(
            padding=(1, 1)
        )
    )

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation='relu',
            name='conv6'
        )
    )

    model.add(
        ZeroPadding2D(
            padding=(1, 1)
        )
    )

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation='relu',
            name='conv7'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        )
    )

    model.add(Flatten())

    model.add(
        Dense(
            1024,
            activation='relu',
            name='dense1'
        )
    )

    model.add(Dropout(0.5))

    model.add(
        Dense(
            512,
            activation='relu',
            name='dense2'
        )
    )

    model.add(Dropout(0.5))

    model.add(
        Dense(
            6,
            activation='softmax',
            name='dense3'
        )
    )

    print("CPU-compatible model created.")

    # ========================================================
    # LOAD OLD H5 WEIGHTS DIRECTLY
    # ========================================================

    if weights_path:

        print("Reading old H5 weights directly...")

        with h5py.File(weights_path, 'r') as f:

            # ------------------------------------------------
            # Convolution weights
            #
            # H5 format:
            # (OUT, IN, H, W)
            #
            # TensorFlow format:
            # (H, W, IN, OUT)
            # ------------------------------------------------

            conv_names = [
                'convolution2d_1',
                'convolution2d_2',
                'convolution2d_3',
                'convolution2d_4',
                'convolution2d_5',
                'convolution2d_6',
                'convolution2d_7'
            ]

            model_conv_names = [
                'conv1',
                'conv2',
                'conv3',
                'conv4',
                'conv5',
                'conv6',
                'conv7'
            ]

            for old_name, new_name in zip(
                conv_names,
                model_conv_names
            ):

                kernel_path = (
                    old_name +
                    '/' +
                    old_name +
                    '_W'
                )

                bias_path = (
                    old_name +
                    '/' +
                    old_name +
                    '_b'
                )

                kernel = np.array(
                    f[kernel_path]
                )

                bias = np.array(
                    f[bias_path]
                )

                print(
                    old_name,
                    "H5:",
                    kernel.shape
                )

                # OIHW -> HWIO
                kernel = np.transpose(
                    kernel,
                    (2, 3, 1, 0)
                )

                layer = model.get_layer(new_name)

                print(
                    new_name,
                    "TensorFlow:",
                    kernel.shape
                )

                layer.set_weights([
                    kernel.astype(np.float32),
                    bias.astype(np.float32)
                ])

            # ------------------------------------------------
            # Dense layers
            #
            # H5 format is already:
            # (input, output)
            # ------------------------------------------------

            dense_names = [
                'dense1',
                'dense2',
                'dense3'
            ]

            old_dense_names = [
                'dense_1',
                'dense_2',
                'dense_3'
            ]

            for old_name, new_name in zip(
                old_dense_names,
                dense_names
            ):

                weight_path = (
                    old_name +
                    '/' +
                    old_name +
                    '_W'
                )

                bias_path = (
                    old_name +
                    '/' +
                    old_name +
                    '_b'
                )

                weights = np.array(
                    f[weight_path]
                )

                bias = np.array(
                    f[bias_path]
                )

                print(
                    old_name,
                    "weights:",
                    weights.shape
                )

                model.get_layer(new_name).set_weights([
                    weights.astype(np.float32),
                    bias.astype(np.float32)
                ])

        print("ALL OLD WEIGHTS CONVERTED SUCCESSFULLY.")

    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model