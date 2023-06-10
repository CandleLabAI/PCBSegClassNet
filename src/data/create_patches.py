# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# --------------------------------------------------------------------

"""
Create RGB patches from images and masks

Usage:
python create_patches.py -i ../../data/segmentation/images/ -m ../../data/segmentation/masks -cd ../../data/classification/images/ -ps 768
"""

from glob import glob
import argparse
import shutil
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

def create_patches(images_list,
                   masks_list,
                   patch_size,
                   dest_images_dir,
                   dest_masks_dir):
    """
    Helper function to create patches
    Args:
        images_list: list of image paths
        masks_list: list of mask paths
        patch_size: integer size of patch
        dest_images_dir: destination directory to store images
        dest_masks_dir: destination directory to store masks
    """
    count = 0
    with tqdm(total=len(images_list)) as pbar:
        for image, mask in zip(images_list, masks_list):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msk = cv2.imread(mask)
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)

            for j in range(0, img.shape[0], patch_size):
                for k in range(0, img.shape[1], patch_size):
                    temp = img[j:j+patch_size, k:k+patch_size, :]
                    if temp.shape[0] != patch_size:
                        j = j - (patch_size - temp.shape[0])
                        temp = img[j:j+patch_size, k:k+patch_size, :]
                    if temp.shape[1] != patch_size:
                        k = k - (patch_size - temp.shape[1])
                        temp = img[j:j+patch_size, k:k+patch_size, :]
                    cv2.imwrite(os.path.join(dest_images_dir,
                                             f"image_{count}.png"),
                                             cv2.cvtColor(img[j:j+patch_size, k:k+patch_size, :],
                                                          cv2.COLOR_BGR2RGB))
                    cv2.imwrite(os.path.join(dest_masks_dir,
                                             f"image_{count}.png"),
                                             cv2.cvtColor(msk[j:j+patch_size, k:k+patch_size, :],
                                                          cv2.COLOR_BGR2RGB))
                    count += 1
            pbar.update(1)

def split_data(source_images_dir,
               source_masks_dir,
               source_crops_dir,
               dest_images_dir,
               dest_masks_dir,
               dest_crops_dir,
               split_percentage):
    """
    Helper function to move patches and crop data into train and val
    """
    images_list = sorted([os.path.join(source_images_dir, image)
                          for image in os.listdir(source_images_dir)])
    masks_list = sorted([os.path.join(source_masks_dir, image)
                         for image in os.listdir(source_masks_dir)])
    crops_list = sorted(glob(os.path.join(source_crops_dir, "*/*")))
    crops_labels_list = [crop.split(os.path.sep)[-2] for crop in crops_list]

    train_images, test_images, train_masks, test_masks = train_test_split(images_list,
                                                        masks_list,
                                                        test_size=split_percentage,
                                                        random_state=42)

    train_crops, test_crops, train_labels, test_labels = train_test_split(crops_list,
                                                        crops_labels_list,
                                                        test_size=split_percentage,
                                                        random_state=42)

    if not os.path.exists(os.path.join(dest_images_dir, "train", "images")):
        os.makedirs(os.path.join(dest_images_dir, "train", "images"))
    if not os.path.exists(os.path.join(dest_images_dir, "train", "masks")):
        os.makedirs(os.path.join(dest_images_dir, "train", "masks"))

    if not os.path.exists(os.path.join(dest_images_dir, "val", "images")):
        os.makedirs(os.path.join(dest_images_dir, "val", "images"))
    if not os.path.exists(os.path.join(dest_images_dir, "val", "masks")):
        os.makedirs(os.path.join(dest_images_dir, "val", "masks"))

    if not os.path.exists(os.path.join(dest_crops_dir, "train")):
        os.makedirs(os.path.join(dest_crops_dir, "train"))
    if not os.path.exists(os.path.join(dest_crops_dir, "val")):
        os.makedirs(os.path.join(dest_crops_dir, "val"))

    for image, mask in zip(train_images, train_masks):
        image_name = os.path.basename(image)
        mask_name = os.path.basename(mask)
        shutil.move(image, os.path.join(dest_images_dir, "train", "images", image_name))
        shutil.move(mask, os.path.join(dest_masks_dir, "train", "masks", mask_name))

    for image, mask in zip(test_images, test_masks):
        image_name = os.path.basename(image)
        mask_name = os.path.basename(mask)
        shutil.move(image, os.path.join(dest_images_dir, "val", "images", image_name))
        shutil.move(mask, os.path.join(dest_masks_dir, "val", "masks", mask_name))

    for image, label in zip(train_crops, train_labels):
        image_name = os.path.basename(image)
        if not os.path.exists(os.path.join(dest_crops_dir, "train", label)):
            os.makedirs(os.path.join(dest_crops_dir, "train", label))
        shutil.move(image, os.path.join(dest_crops_dir, "train", label, image_name))

    for image, label in zip(test_crops, test_labels):
        image_name = os.path.basename(image)
        if not os.path.exists(os.path.join(dest_crops_dir, "val", label)):
            os.makedirs(os.path.join(dest_crops_dir, "val", label))
        shutil.move(image, os.path.join(dest_crops_dir, "val", label, image_name))

def main(images_dir,
         masks_dir,
         crops_dir,
         patch_size=768):
    """
    Main function for creating patches and splitting data
    """
    dest_images_dir = os.path.join(os.path.abspath(os.path.join(images_dir, os.pardir)),
                                   "patches", "images")
    dest_masks_dir = os.path.join(os.path.abspath(os.path.join(masks_dir, os.pardir)),
                                  "patches", "masks")

    images_list = sorted([os.path.join(images_dir, image) for image in os.listdir(images_dir)])
    masks_list = sorted([os.path.join(masks_dir, image) for image in os.listdir(masks_dir)])

    # create directory if not exist
    if not os.path.exists(dest_images_dir):
        os.makedirs(dest_images_dir)
    if not os.path.exists(dest_masks_dir):
        os.makedirs(dest_masks_dir)

    create_patches(images_list,
                   masks_list,
                   patch_size,
                   dest_images_dir,
                   dest_masks_dir)

    source_images_dir = dest_images_dir
    source_masks_dir = dest_masks_dir
    source_crops_dir = crops_dir
    dest_images_dir = os.path.abspath(os.path.join(images_dir, os.pardir))
    dest_masks_dir = os.path.abspath(os.path.join(masks_dir, os.pardir))
    dest_crops_dir = os.path.abspath(os.path.join(crops_dir, os.pardir))

    split_data(source_images_dir,
               source_masks_dir,
               source_crops_dir,
               dest_images_dir,
               dest_masks_dir,
               dest_crops_dir,
               split_percentage=0.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PCBSegClassNet')
    parser.add_argument('-i',
                        "--images_dir",
                        type=str,
                        required=True,
                        help="The path of directory containing input images")
    parser.add_argument('-m',
                        '--masks_dir',
                        type=str,
                        required=True,
                        help="The path of directory containing annotations")
    parser.add_argument('-cd',
                        '--crops_dir',
                        type=str,
                        required=True,
                        help="The path of directory containing crops images")
    parser.add_argument('-ps',
                        '--patch_size',
                        type=int,
                        required=True,
                        help="The patch size for creatig patches of input data")
    args = parser.parse_args()

    main(images_dir = args.images_dir,
         masks_dir = args.masks_dir,
         crops_dir = args.crops_dir,
         patch_size=args.patch_size)
