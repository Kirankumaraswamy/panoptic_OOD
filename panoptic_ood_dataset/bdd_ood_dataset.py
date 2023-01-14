#!/usr/bin/python
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np
import math
import cv2
from skimage import exposure, img_as_float

# Image processing
from PIL import Image

# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.helpers.labels import id2label, labels, trainId2label
from pycocotools.coco import COCO as coco_tools
import random

coco_json_data = json.load(open("/home/kumarasw/OOD_dataset/temp_bdd/bdd100k/labels/pan_seg/polygons/val/pan_seg_val.json"))

colormasks = np.array(Image.open("/home/kumarasw/OOD_dataset/temp_bdd/bdd100k/labels/pan_seg/colormaps/val/7d2f7975-e0c1c5a7.png"))
bit_masks = np.array(Image.open("/home/kumarasw/OOD_dataset/temp_bdd/bdd100k/labels/pan_seg/bitmasks/val/7d2f7975-e0c1c5a7.png"))
pan_mask = np.array(Image.open("/home/kumarasw/OOD_dataset/bdd/temp/val/7d2f7975-e0c1c5a7.png"))

bdd_jsondata = json.load(open("/home/kumarasw/OOD_dataset/bdd/bdd/gtFine/bdd_panoptic.json"))
cityscapes_json_data = json.load(open("/home/kumarasw/OOD_dataset/cityscapes_ood/gtFine/cityscapes_panoptic_train.json"))
a = 2