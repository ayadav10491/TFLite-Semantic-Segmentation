"""
The implementation of some utils.

@Author: Akash Yadav
@Github: https://github.com/ayadav10491
@Project: https://github.com/ayadav10491/TFLite-Semantic-Segmentation


"""
from keras_preprocessing import image as keras_image
from PIL import Image
import numpy as np
import cv2


def load_image(name):
    image= Image.open(name)
    #print(img)
    #print('Load Image', name, np.array(img).shape)
    return np.array(image)


def resize_image(image, label, desired_size=None):
    if desired_size is not None:
        image = cv2.resize(image, dsize=desired_size[::-1])
        label = cv2.resize(label, dsize=desired_size[::-1], interpolation=cv2.INTER_NEAREST)
    return image, label


def one_hot(label, num_classes):
    #print(label.shape)
    if np.ndim(label) == 3:
        label = np.squeeze(label, axis=-1)
    assert np.ndim(label) == 2
    semantic_map = np.ones(shape=label.shape[0:2] + (num_classes))
    for i in range(num_classes):
        semantic_map[:, :, i] = np.equal(label, i).astype('float32')
    return semantic_map


def decode_one_hot(one_hot_seg_map):
    return np.argmax(one_hot_seg_map, axis=-1)





