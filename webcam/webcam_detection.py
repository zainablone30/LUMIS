import argparse
import sys

sys.path.append("../")

import cv2
import numpy as np

import face_detection_utilities as fdu
import model.myVGG as vgg


windowsName = 'Preview Screen'

parser = argparse.ArgumentParser(
    description='A live emotion recognition from webcam'
)

parser.add_argument(
    '-testImage',
    help='Given the path of testing image, the program will predict the result of the image.'
)

args = parser.parse_args()

FACE_SHAPE = (48, 48)


# Load trained model
model = vgg.VGG_16('my_model_weights_83.h5')


# Emotion labels
emo = [
    'Angry',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]


def refreshFrame(frame, faceCoordinates):

    if faceCoordinates is not None:
        fdu.drawFace(frame, faceCoordinates)

    cv2.imshow(windowsName, frame)


def showScreenAndDectect(capture):

    while True:

        flag, frame = capture.read()

        if not flag:
            print("Failed to read frame from camera.")
            break

        faceCoordinates = fdu.getFaceCoordinates(frame)

        refreshFrame(frame, faceCoordinates)

        if faceCoordinates is not None:

            face_img = fdu.preprocess(
                frame,
                faceCoordinates,
                face_shape=FACE_SHAPE
            )

            # ------------------------------------------------
            # Model expects:
            # (batch, height, width, channel)
            #
            # face_img:
            # (48, 48)
            #
            # -> (48, 48, 1)
            # -> (1, 48, 48, 1)
            # ------------------------------------------------

            input_img = np.expand_dims(
                face_img,
                axis=-1
            )

            input_img = np.expand_dims(
                input_img,
                axis=0
            )

            input_img = input_img.astype(
                np.float32
            )

            # Predict
            result = model.predict(
                input_img,
                verbose=0
            )[0]

            index = np.argmax(result)

            print(
                emo[index],
                'prob:',
                float(result[index])
            )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def getCameraStreaming():

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():

        print("Failed to capture video streaming")
        sys.exit(1)

    print("Successed to capture video streaming")

    return capture


def main():

    print("Enter main() function")

    capture = getCameraStreaming()

    cv2.startWindowThread()

    cv2.namedWindow(
        windowsName,
        cv2.WINDOW_NORMAL
    )

    showScreenAndDectect(capture)


if __name__ == '__main__':
    main()