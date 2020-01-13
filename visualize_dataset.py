import glob
import random
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

import sys
sys.path.append('drive/My Drive/NNDS/project/')
from data_loader import get_pairs_from_paths, get_triplet_from_paths, DATA_LOADER_SEED, class_colors, DataLoaderError

random.seed(DATA_LOADER_SEED)

def _get_colored_segmentation_image(img, seg, colors, n_classes, do_augment=False):
    """ Return a colored segmented image """
    seg_img = np.zeros_like(seg)

    if do_augment:
        img, seg[:, :, 0] = augment_seg(img, seg[:, :, 0])

    for c in range(n_classes):
        seg_img[:, :, 0] += ((seg[:, :, 0] == c) *
                            (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg[:, :, 0] == c) *
                            (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg[:, :, 0] == c) *
                            (colors[c][2])).astype('uint8')

    return img , seg_img


def visualize_segmentation_dataset(images_path, segs_path, n_classes,
                                   do_augment=False, ignore_non_matching=False,
                                   no_show=False):
    # Get image-segmentation pairs
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path,
                        ignore_non_matching=ignore_non_matching)

    # Get the colors for the classes
    colors = class_colors

    print("Please press any key to display the next image")
    for im_fn, seg_fn in img_seg_pairs:
        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)
        print("Found the following classes in the segmentation image:", np.unique(seg))
        img , seg_img = _get_colored_segmentation_image(img, seg, colors, n_classes, do_augment=do_augment)
        print("Please press any key to display the next image")
        cv2_imshow(img)
        cv2_imshow(seg_img)
        cv2.waitKey()



def visualize_segmentation_dataset_one(images_path, segs_path, n_classes,
                                       do_augment=False, no_show=False, ignore_non_matching=False,
return_images = False):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path, ignore_non_matching=ignore_non_matching)

    colors = class_colors

    im_fn, seg_fn = random.choice(img_seg_pairs)

    img = cv2.imread(im_fn)
    seg = cv2.imread(seg_fn)
    print("Found the following classes in the segmentation image:", np.unique(seg))

    img,seg_img = _get_colored_segmentation_image(img, seg, colors,n_classes, do_augment=do_augment)

    if not no_show:
        cv2_imshow(img)
        cv2_imshow(seg_img)
        cv2.waitKey()
    
    if return_images:
        return img, seg_img
    
def visualize_results(images_path, segs_path, res_path, n_classes,
                      do_augment=False, no_show=False, ignore_non_matching=False,
                      return_images = False):

    img_seg_pairs = get_triplet_from_paths(images_path, segs_path, res_path, ignore_non_matching=ignore_non_matching)

    colors = class_colors

    im_fn, seg_fn, res_fn = random.choice(img_seg_pairs)

    img = cv2.imread(im_fn)
    seg = cv2.imread(seg_fn)
    res = cv2.imread(res_fn)
    print("Found the following classes in the segmentation image:", np.unique(seg))

    img,seg_img = _get_colored_segmentation_image(img, seg, colors,n_classes, do_augment=do_augment)

    if not no_show:
        cv2_imshow(img)
        cv2_imshow(seg_img)
        cv2_imshow(res)
        cv2.waitKey()
    
    if return_images:
        return img, seg_img, res
