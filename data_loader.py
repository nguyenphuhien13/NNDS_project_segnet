import glob
import itertools
import os
import random
import six
import numpy as np
import json
import cv2
from tqdm import tqdm_notebook as tqdm
from google.colab.patches import cv2_imshow
import warnings
warnings.filterwarnings('ignore')

IMAGE_ORDERING = 'channels_last'
DATA_LOADER_SEED = 0

random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


def get_image_array(image_input, width, height):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif  isinstance(image_input, six.string_types)  :
        img = cv2.imread(image_input, 1)

    # resize image
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]

    return img

def get_segmentation_array(image_input, nClasses, width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types) :
        img = cv2.imread(image_input, 1)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels

def data_generator(im_path, annot_path, batch_size,
                  n_classes, height, width):

    images = glob.glob(im_path + "*.jpg") + glob.glob(im_path + "*.png") + glob.glob(im_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(annot_path + "*.jpg") + glob.glob(annot_path + "*.png") + glob.glob(annot_path + "*.jpeg")
    segmentations.sort()

    data = []
    for i, j in zip(images,segmentations):
        data.append((i,j))
    
    random.shuffle(data)
    zipped = itertools.cycle(data)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            X.append(get_image_array(im, width, height))
            Y.append(get_segmentation_array(seg, n_classes, width, height))

        yield np.array(X), np.array(Y)
