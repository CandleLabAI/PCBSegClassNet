# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# --------------------------------------------------------------------
import sys
sys.path.append("../models")
from edsr import make_model
import albumentations as A
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import argparse
import shutil
import cv2
import ast
import os

# color_values is used to encode mask into one hot mapping.
color_values = {
    "R": (255, 0, 0),
    "C": (255, 255, 0),
    "U": (0, 234, 255), 
    "Q": (170, 0, 255),
    "J": (255, 127, 0),
    "L": (191, 255, 0),
    "RA": (0, 149, 255),
    "D": (106, 255, 0),
    "RN": (0, 64, 255),
    "TP": (237, 185, 185),
    "IC": (185, 215, 237),
    "P": (231, 233, 185),
    "CR": (220, 185, 237),
    "M": (185, 237, 224),
    "BTN": (143, 35, 35),
    "FB": (35, 98, 143),
    "CRA": (143, 106, 35),
    "SW": (107, 35, 143),
    "T": (79, 143, 35),
    "F": (115, 115, 115),
    "V": (204, 204, 204),
    "LED": (245, 130, 48),
    "S": (220, 190, 255),
    "QA": (170, 255, 195),
    "JP": (255, 250, 200)
}

def prepare_data(source_image_dir, source_annotation_dir, dest_images_dir, dest_masks_sir, crops_dest_dir, model):
    annotations_list = glob(os.path.join(source_annotation_dir, "*.csv"))
    count = 0
    cnt = 0
    transform = A.Compose([
        A.augmentations.transforms.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1.0)
    ])
    with tqdm(total=len(annotations_list)) as pbar:
        for annotation in annotations_list:
            df = pd.read_csv(annotation)
            # checking if atleast 1 designation is present in annotation
            if df["Designator"].isna().sum() != df.shape[0]:
                image_name = list(df["Image File"].unique())
                if os.path.exists(os.path.join(source_image_dir, image_name[0])):
                    img = cv2.imread(os.path.join(source_image_dir, image_name[0]))
                    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                    mask = np.zeros(shape=img.shape, dtype=np.uint8)
                    transformed = transform(image=img1, mask=mask)
                    img1 = transformed['image']
                    vertices_list = list(df["Vertices"])
                    designator_list = list(df["Designator"])
                    for (anote, cat) in zip(vertices_list, designator_list):
                        if cat in color_values:
                            color_code = color_values[cat]
                        else:
                            continue
                        try:
                            pts = np.array(ast.literal_eval(anote))[0].reshape((-1, 1, 2))
                        except:
                            continue
                            # pts = np.array(np.array(ast.literal_eval(anote))[0]).reshape((-1, 1, 2))
                        mask = cv2.polylines(mask, [pts], True, color_code, 2)
                        mask = cv2.fillPoly(mask, [pts], color=color_code)
                        # create crops
                        mask_copy = np.zeros(shape=img.shape, dtype=np.uint8)
                        mask_copy = cv2.polylines(mask_copy, [pts], True, color_code, 2)
                        mask_copy = cv2.fillPoly(mask_copy, [pts], color=color_code)
                        mask_copy = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
                        contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        x,y,w,h = cv2.boundingRect(contours[0])
                        comp_img = img[y:y+h, x:x+w]
                        comp_img = tf.image.resize(comp_img, (150, 150))
                        preds = model.predict_step(comp_img)
                        comp_img = np.array(preds)
                        if not os.path.exists(os.path.join(crops_dest_dir, cat)):
                            os.makedirs(os.path.join(crops_dest_dir, cat))
                        cv2.imwrite(os.path.join(crops_dest_dir, cat, f"image_{cnt}.png"), cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB))
                        cnt += 1
                        
                    if len(np.unique(mask)) > 1:
                        cv2.imwrite(os.path.join(dest_masks_sir, f"image_{count}.png"), cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                        # shutil.copy(os.path.join(source_image_dir, image_name[0]), os.path.join(dest_images_dir, f"image_{count}.png"))
                        cv2.imwrite(os.path.join(dest_images_dir, f"image_{count}.png"), cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                    count+=1
                    pbar.update(1)

def main(source_image_dir, source_annotation_dir, dest_images_dir, dest_masks_sir, dest_crops_dir, model):
    # create dest directory if not exist
    if not os.path.exists(dest_images_dir):
        os.makedirs(dest_images_dir)

    if not os.path.exists(dest_masks_sir):
        os.makedirs(dest_masks_sir)
    
    if not os.path.exists(dest_crops_dir):
        os.makedirs(dest_crops_dir)

    prepare_data(source_image_dir, source_annotation_dir, dest_images_dir, dest_masks_sir, dest_crops_dir, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PCBSegClassNet')
    parser.add_argument('-i',
                        "--images_dir",
                        type=str,
                        required=True,
                        help="The path of directory containing input images")
    parser.add_argument('-a',
                        '--annotations_dir',
                        type=str,
                        required=True,
                        help="The path of directory containing annotations")
    parser.add_argument('-id',
                        '--images_dest_dir',
                        type=str,
                        required=True,
                        help="The path of destination directory where images needs to be stored")
    parser.add_argument('-ad',
                        '--annotations_dest_dir',
                        type=str,
                        required=True,
                        help="The path of destination directory where masks needs to be stored")
    parser.add_argument('-cd',
                        '--crops_dest_dir',
                        type=str,
                        required=True,
                        help="The path of destination directory where crops needs to be stored")
    args = parser.parse_args()

    model = make_model(num_filters=64, num_of_residual_blocks=16)
    model.load_weights("../../checkpoints/super_resolution.h5")
    
    main(source_image_dir = args.images_dir,
         source_annotation_dir = args.annotations_dir,
         dest_images_dir = args.images_dest_dir,
         dest_masks_sir = args.annotations_dest_dir,
         dest_crops_dir = args.crops_dest_dir,
         model = model)