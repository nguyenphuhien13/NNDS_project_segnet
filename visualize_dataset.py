import glob
import random
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

import sys
sys.path.append('drive/My Drive/NNDS/project/')
from data_loader import DATA_LOADER_SEED, class_colors

random.seed(DATA_LOADER_SEED)
cclass_colors

def visualize_dataset(images_path, segs_path, n_classes, show_all = False, num_load = 2, colors = colors):

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    if show_all:
        for im_fn, seg_fn in zip(images, segmentations):

            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)

            seg_img = np.zeros_like(seg)

            for c in range(n_classes):
                seg_img[:, :, 0] += ((seg[:, :, 0] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((seg[:, :, 0] == c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((seg[:, :, 0] == c) * (colors[c][2])).astype('uint8')

            cv2_imshow(img)
            cv2_imshow(seg_img)
    
    else:
        fig, m_axs = plt.subplots(num_load, 2, figsize = (16, 16))

        for (ax1, ax2) in m_axs:

            index = random.randrange(len(images))
            im_fn, seg_fn = images[index], segmentations[index]

            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)

            seg_img = np.zeros_like(seg)

            for c in range(n_classes):
                seg_img[:, :, 0] += ((seg[:, :, 0] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((seg[:, :, 0] == c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((seg[:, :, 0] == c) * (colors[c][2])).astype('uint8')

            ax1.imshow(img)
            ax2.imshow(seg_img)

def visualize_results(images_path, segs_path, results_path, n_classes, show_all = False, num_load = 2, colors = colors):

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()
    results = glob.glob(results_path + "*.jpg") + glob.glob(results_path + "*.png") + glob.glob(results_path + "*.jpeg")
    results.sort()

    if show_all:
        for im_fn, seg_fn, res_fn in zip(images, segmentations, results):
            assert (im_fn.split('/')[-1].split('.')[0] == seg_fn.split('/')[-1].split('.')[0])
            assert (im_fn.split('/')[-1].split('.')[0] == res_fn.split('/')[-1].split('.')[0])

            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            res = cv2.imread(res_fn)

            seg_img = np.zeros_like(seg)

            for c in range(n_classes):
                seg_img[:, :, 0] += ((seg[:, :, 0] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((seg[:, :, 0] == c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((seg[:, :, 0] == c) * (colors[c][2])).astype('uint8')

            cv2_imshow(img)
            cv2_imshow(seg_img)
            cv2_imshow(res)
    
    else:
        fig, m_axs = plt.subplots(num_load, 3, figsize = (16, 16))
        for (ax1, ax2, ax3) in m_axs:

            index = random.randrange(len(images))
            im_fn, seg_fn, res_fn = images[index], segmentations[index], results[index]

            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            res = cv2.imread(res_fn)

            seg_img = np.zeros_like(seg)

            for c in range(n_classes):
                seg_img[:, :, 0] += ((seg[:, :, 0] == c) * (colors[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((seg[:, :, 0] == c) * (colors[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((seg[:, :, 0] == c) * (colors[c][2])).astype('uint8')

            ax1.imshow(img)
            ax2.imshow(seg_img)
            ax3.imshow(res)
