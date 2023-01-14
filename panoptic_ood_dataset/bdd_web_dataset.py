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
from extract_coco_instances import coco_categories

ood_train_id = 19
ood_id = 50

city_height = 1024
city_width = 2048
bins = np.linspace(0, 100, 12)
border_smooth_bin = np.linspace(0, 200000, 5)
random_no_instances = 3
# total number of samples to generate for each cityscapes image
sample_instances = 9

web_categories = {
    "dog": ("road", "car", 0.2, "test"),
    "horse": ("road", "car", 1, "test"),
    "buffallo": ("road", "car", 1, "test"),
    "lion": ("road", "car", 0.5, "test"),
    "sheep": ("road", "car", 0.4, "test"),
    "elephant": ("road", "car", 2, "test"),
    "deer": ("road", "car", 0.4, "test"),
    "ostritch": ("road", "car", 0.5, "test"),
    "hen": ("road", "car", 0.05, "test"),
    "ducks": ("road", "car", 0.05, "test"),
    "trash bin": ("road", "car", 0.3, "test"),
    "brick": ("road", "car", 0.05, "test"),
    "suitcase": ("road", "car", 0.2, "test"),
    "chair": ("road", "car", 0.3, "test"),
    "ball": ("road", "car", 0.05, "test"),
    "umbrella": ("road", "car", 0.3, "test"),
    "bottle": ("road", "car", 0.05, "test"),
    "helmet": ("road", "car", 0.05, "test"),
    "rickshaw": ("road", "car", 1, "test"),
    "wooden box": ("road", "car", 0.2, "test"),
    "road blocker": ("road", "car", 0.3, "test"),
    "baby stroller": ("road", "car", 0.3, "test"),
    "tyre": ("road", "car", 0.2, "test"),
    "speaker": ("road", "car", 0.2, "test"),
    "barrel": ("road", "car", 0.4, "test"),
    "fire extinguisher": ("road", "car", 0.15, "test"),
    "buldozer": ("road", "car", 2, "test"),
    "trolley": ("road", "car", 0.3, "test"),
    "bird": ("road", "sky", 0.05, "test"),
    "bear": ("road", "car", 0.5, "test"),
    "cow": ("road", "car", 1, "test"),
    "tiger": ("road", "car", 0.5, "test"),
    "fire hydrant": ("road", "car", 0.2, "test"),
    "portable public toilet": ("road", "car", 1, "test"),
    "drone": ("road", "sky", 0.1, "test"),
    "skating board": ("road", "car", 0.15, "test"),
    "wolf": ("road", "car", 0.4, "test"),
    "cat": ("road", "car", 0.1, "test"),
    "hand bag": ("road", "car", 0.15, "test"),
    "barricade": ("road", "car", 0.6, "test"),
    "shoe": ("road", "car", 0.05, "test"),
    "rucksack": ("road", "car", 0.2, "test"),
    "lawn mover": ("road", "car", 0.3, "test"),
    "robot": ("road", "car", 0.4, "test"),
    "wooden log": ("road", "car", 0.2, "test"),
    "sofa chair": ("road", "car", 0.8, "test"),
    "boat": ("road", "car", 0.8, "test"),
    "giraffe": ("road", "car", 2, "test"),
    "tortoise": ("road", "car", 0.1, "test")
    }

categories_test = [
         "dog", "horse", "buffallo", "lion", "sheep", "black panther", "deer", "ostritch", "hen", "ducks",
        "trash bin", "brick", "suitcase", "chair", "ball", "umbrella", "bottle", "helmet", "rickshaw", "wooden box",
        "road blockers", "baby stroller", "tyre", "construction items", "speaker", "barrel",  "fire extinguisher",  "buldozer", "trolley"
    ]


def apply_blending(mask, bdd_web_panoptic_values, bdd_depth_img, bdd_masked_image,
                   bdd_original):
    # Image.fromarray(bdd_masked_image).save(os.path.join("/home/kumarasw/kiran/bdd_ood", "no_blending.png"))

    # apply smoothing of ood object at border

    # Load original image and find contours.
    # Blur the original image and save it in a different variable.
    # Create an empty mask and draw the detected contours on it.
    # Use np.where() method to select the pixels from the mask (contours) where you want blurred values and then replace it.

    for instance_id in np.unique(bdd_web_panoptic_values):
        if instance_id == 0:
            continue

        smooth_mask = np.zeros((city_height, city_width), dtype="uint8")
        ood_pixels = np.where(bdd_web_panoptic_values == instance_id)
        smooth_mask[ood_pixels] = 255
        # set border smoothness thickness based on the object size
        bin_s = np.digitize(len(ood_pixels[0]), border_smooth_bin)

        blurred_img = cv2.blur(bdd_masked_image, (5, 5))
        contour_mask = np.zeros(blurred_img.shape, np.uint8)

        contours, hierarchy = cv2.findContours(image=smooth_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(image=contour_mask, contours=contours, contourIdx=-1, color=(255, 255, 255),
                         thickness=bin_s + 1,
                         lineType=cv2.LINE_AA)

        bdd_masked_image = np.where(contour_mask == np.array([255, 255, 255]), blurred_img,
                                           bdd_masked_image)

        # Image.fromarray(contour_mask).save(os.path.join("/home/kumarasw/kiran/bdd_web", "contours.png"))
        # print("border size: ", bin_s)
    # Image.fromarray(bdd_masked_image).save(os.path.join("/home/kumarasw/kiran/bdd_web", "border_smoothed.png"))

    # Radial motion blur implementation
    # generating the kernel
    panoptic_ids = np.unique(bdd_web_panoptic_values).tolist()
    panoptic_ids.remove(0)
    motion_mask = np.zeros((city_height, city_width, 3), dtype="uint8")

    # vertical motion blur implementation
    # generating the kernel
    size = random.randint(2, 5)
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    for id in panoptic_ids:
        mask = np.where(bdd_web_panoptic_values == id)
        area_mask = np.sum(bdd_web_panoptic_values == id)
        if area_mask > 100:
            x_min = np.min(mask[0])
            x_max = np.max(mask[0])
            y_min = np.min(mask[1])
            y_max = np.max(mask[1])

            web_mask = bdd_masked_image[x_min:x_max, y_min:y_max, :]
            web_mask = img_as_float(web_mask)
            img_out = web_mask.copy()

            row, col, channel = web_mask.shape

            xx = np.arange(col)
            yy = np.arange(row)

            x_mask = np.tile(xx, (row, 1))
            y_mask = np.tile(yy, (col, 1))
            y_mask = np.transpose(y_mask)

            center_y = (row - 1) / 2.0
            center_x = (col - 1) / 2.0

            R = np.sqrt((x_mask - center_x) ** 2 + (y_mask - center_y) ** 2)

            angle = np.arctan2(y_mask - center_y, x_mask - center_x)

            Num = random.randint(2, 5)
            arr = np.arange(Num)

            R_arr = np.moveaxis(np.array([R] * len(arr)), 0, -1) - arr
            R_arr[R_arr < 0] = 0

            new_x = R_arr * np.cos(np.moveaxis(np.array([angle] * len(arr)), 0, -1)) + center_x
            new_y = R_arr * np.sin(np.moveaxis(np.array([angle] * len(arr)), 0, -1)) + center_y

            int_x = new_x.astype(int)
            int_y = new_y.astype(int)

            int_x[int_x > col - 1] = col - 1
            int_x[int_x < 0] = 0
            int_y[int_y < 0] = 0
            int_y[int_y > row - 1] = row - 1

            img_out[:, :, 0] = web_mask[int_y, int_x, 0].sum(axis=-1) / Num
            img_out[:, :, 1] = web_mask[int_y, int_x, 1].sum(axis=-1) / Num
            img_out[:, :, 2] = web_mask[int_y, int_x, 2].sum(axis=-1) / Num

            img_out = img_out * 255
            img_out = img_out.astype(np.uint8)

            # applying the kernel to the input image
            motion_blur = cv2.filter2D(img_out, -1, kernel_motion_blur)

            motion_mask[x_min:x_max, y_min:y_max, :] = motion_blur
            bdd_masked_image[mask] = motion_mask[mask]

    # Image.fromarray(bdd_masked_image).save(os.path.join("/home/kumarasw/kiran/bdd_web", "radial_motion__blur.png"))

    # apply inverse color histogram
    bdd_masked_image = exposure.match_histograms(bdd_masked_image, bdd_original, channel_axis=-1)

    # Image.fromarray(bdd_masked_image).save( os.path.join("/home/kumarasw/kiran/bdd_web", "exposure_matching.png"))

    '''# inverse histogram only for patch of web image
    panoptic_ids = np.unique(bdd_web_panoptic_values).tolist()
    panoptic_ids.remove(0)
    exposure_mask = np.zeros((city_height, city_width, 3), dtype="uint8")
    for id in panoptic_ids:
        mask = np.where(bdd_web_panoptic_values==id)
        x_min = np.min(mask[0])
        x_max = np.max(mask[0])
        y_min = np.min(mask[1])
        y_max = np.max(mask[1])

        web_mask = bdd_masked_image[x_min:x_max, y_min:y_max, :]
        city_mask = bdd_original[x_min:x_max, y_min:y_max, :]

        #Image.fromarray(web_mask).save(os.path.join("/home/kumarasw/kiran/bdd_web", "web_mask.png"))

        #Image.fromarray(city_mask).save(os.path.join("/home/kumarasw/kiran/bdd_web", "city_mask.png"))

        exp_image = exposure.match_histograms(web_mask, bdd_original, channel_axis=-1)

        exposure_mask[x_min:x_max, y_min:y_max, :] = exp_image
        bdd_masked_image[mask] = exposure_mask[mask]

    bdd_masked_image = exposure.match_histograms(bdd_masked_image, bdd_original, channel_axis=-1)'''

    # Image.fromarray(bdd_masked_image).save( os.path.join("/home/kumarasw/kiran/bdd_web", "exposure_matching.png"))

    # Apply brighness to OOD pixels to match bdd pixels
    hsv_web = cv2.cvtColor(bdd_masked_image, cv2.COLOR_RGB2HSV)
    for instance_id in np.unique(bdd_web_panoptic_values):
        if instance_id == 0:
            continue

        hsv_city = cv2.cvtColor(bdd_original, cv2.COLOR_BGR2HSV)
        V_city_mask = hsv_city[:, :, 2][np.where(bdd_web_panoptic_values == instance_id)]
        avg_V_city_mask = np.mean(V_city_mask)

        hsv_web = cv2.cvtColor(bdd_masked_image, cv2.COLOR_RGB2HSV)
        V_web_mask = hsv_web[:, :, 2][np.where(bdd_web_panoptic_values == instance_id)]
        avg_V_web_mask = np.mean(V_web_mask)

        diff_V = avg_V_city_mask - avg_V_web_mask
        # print(avg_V_city_mask, avg_V_web_mask, diff_V)

        hsv_new = hsv_web[:, :, 2][np.where(bdd_web_panoptic_values == instance_id)] + int(diff_V * 0.9)
        hsv_new[np.where(hsv_new < 0)] = 0
        hsv_new[np.where(hsv_new > 255)] = 255

        hsv_web[:, :, 2][np.where(bdd_web_panoptic_values == instance_id)] = hsv_new

    bdd_masked_image = cv2.cvtColor(hsv_web, cv2.COLOR_HSV2RGB)

    # Image.fromarray(bdd_masked_image).save(os.path.join("/home/kumarasw/kiran/bdd_web", "brightness.png"))

    # apply depth blur at OOD. We use depth values to vary kernel size of Gaussianblur .
    panoptic_ids = np.unique(bdd_web_panoptic_values).tolist()
    panoptic_ids.remove(0)

    for id in panoptic_ids:
        mask = np.where(bdd_web_panoptic_values == id)
        avg_depth = bdd_depth_img[mask].mean()
        bin = np.digitize(avg_depth, bins)
        kernel_size = int((len(bins) - bin) / 3)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size - 1
        blurred_img = cv2.GaussianBlur(bdd_masked_image, (3, 3), 0)
        # print("Kernel_size ",  kernel_size)
        bdd_masked_image[mask] = blurred_img[mask]

    # Image.fromarray(bdd_masked_image).save(os.path.join("/home/kumarasw/kiran/bdd_web", "gaussian_depth_blur.png"))

    # adding color noise
    noise = np.random.randint(0, 4, bdd_masked_image.shape)
    noisy_bdd = bdd_masked_image + noise
    noisy_bdd[np.where(noisy_bdd > 255)] = 255
    bdd_masked_image = noisy_bdd.astype(np.uint8)
    # Image.fromarray(bdd_masked_image).save(os.path.join("/home/kumarasw/kiran/bdd_web", "noise.png"))

    return bdd_masked_image


# The main method
def create_bdd_web_panoptic(args):
    bddPath = args.bddPath
    bddSplit = args.bddSplit
    webInstancePath = args.webInstancePath
    oodSplit = args.oodSplit
    outputFolder = args.outputFolder
    useTrainId = args.useTrainId

    # Where to look for Bdd
    if bddPath is None:
        if 'CITYSCAPES_DATASET' in os.environ:
            bddPath = os.environ['CITYSCAPES_DATASET']
        bddPath = os.path.join(bddPath, "gtFine")

    if bddPath is None:
        if 'COCO_DATASET' in os.environ:
            webPath = os.environ['COCO_DATASET']

    if outputFolder is None:
        raise "Please specify output directory."

    if not os.path.exists(bddPath):
        raise "Specified Bdd path doesn't exist"

    if not os.path.exists(webInstancePath):
        raise "Specified webInstancePath path doesn't exist"

    parent_folder = "bdd_ood"
    mode = "gtFine"

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    web_instances_dir = os.path.join(webInstancePath, oodSplit)
    web_images_folder = "web_rgb"
    web_panoptic_folder = "web_panoptic"
    web_semantic_folder = "web_semantic"

    web_img_Ids = []

    web_json_file = os.path.join(web_instances_dir, "ood_instances.json")
    web_json_data = json.load(open(web_json_file))
    for count, (web_id, value) in enumerate(web_json_data.items()):
        if (os.path.exists(os.path.join(webInstancePath, web_json_data[web_id]["rgb_image"])) and
                os.path.exists(os.path.join(webInstancePath, web_json_data[web_id]["panoptic_image"])) and
                os.path.exists(os.path.join(webInstancePath, web_json_data[web_id]["semantic_image"]))):
            web_img_Ids.append(web_id)
        else:
            print("missing ", web_id)
    web_img_Ids.sort()

    categories = []
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                           'name': label.name,
                           'color': label.color,
                           'supercategory': label.category,
                           'isthing': 1 if label.hasInstances else 0})

    searchFine = os.path.join(bddPath, mode, bddSplit, "*", "*_instanceIds.png")
    # search files

    filesFine = glob.glob(searchFine)
    if bddSplit == "val":
        filesFine.sort(reverse=True)
    else:
        filesFine.sort()

    bdd_files = filesFine

    # randomly add duplicate instances if web instances are less in number
    duplicate_list = []
    random.seed(0)
    random.shuffle(web_img_Ids)
    for c in range(int(2.5 * len(filesFine)) - len(web_img_Ids)):
        rand_index = random.randint(0, len(web_img_Ids)-1)
        duplicate_list.append(web_img_Ids[rand_index])

    web_img_Ids += duplicate_list

    # quit if we did not find anything
    if not bdd_files:
        printError(
            "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(
                bddSplit, searchFine)
        )
    # a bit verbose
    print("Converting {} annotation files for {} set.".format(len(bdd_files), bddSplit))

    # outputBaseFile = os.path.join(outputFolder, "bdd_"+bddSplit+"_web_"+webSplit)
    outputBaseFile = os.path.join(outputFolder, parent_folder, "gtFine", "bdd_panoptic_" + bddSplit)
    outFile = outputBaseFile + ".json"
    labelIdsPath = os.path.join(outputFolder, parent_folder, "gtFine", bddSplit)
    left8bitImgPath = os.path.join(outputFolder, parent_folder, "leftImg8bit", bddSplit)

    if not os.path.isdir(outputBaseFile):
        print("Creating folder {} for saving OOD images and ground truth".format(outputBaseFile))
        os.makedirs(outputBaseFile, exist_ok=True)
    print("Corresponding segmentation ground truth will be saved in {}".format(outFile))
    if not os.path.isdir(left8bitImgPath):
        print("Creating folder {} for saving left8bit images ".format(left8bitImgPath))
        os.makedirs(left8bitImgPath, exist_ok=True)

    status_file_name = "status_" + bddSplit + ".txt"

    # load instance scale json file if exists else create a new one
    # This file contains the average size of the bdd instance at certain distance divided into bins
    scale_output_json = "instance_scale_" + bddSplit + ".json"

    if os.path.exists(os.path.join(outputFolder, parent_folder, scale_output_json)):
        scale_map = json.load(open(os.path.join(outputFolder, parent_folder, scale_output_json)))

    else:
        print(
            "Creating mapping for image size and depth. Please wait as it might take longer time.")
        # map to store the average object size for each class and at given depth(no of bins)
        depth_map = {}
        print("Total images: ", len(bdd_files))
        for i, f in enumerate(bdd_files):
            originalFormat = np.array(Image.open(f))
            fileName = os.path.basename(f)
            bdd_imageId = fileName.replace("_gtFine_instanceIds.png", "")
            bdd_depth_name = fileName.replace("_gtFine_instanceIds.png", "_depth.png")
            city_name = bdd_depth_name.split("_")[0]
            bdd_depth_path = os.path.join(bddPath, "depth", bddSplit, city_name,
                                                     bdd_depth_name)
            bdd_depth_img = np.array(Image.open(bdd_depth_path))

            # Image.fromarray(bdd_depth_img).save(os.path.join(outputFolder, "depth.png"))
            # Image.fromarray(originalFormat).save(os.path.join(outputFolder, "panoptic.png"))

            digitized = np.digitize(bdd_depth_img, bins)

            segmentIds = np.unique(originalFormat)
            for segmentId in segmentIds:
                if segmentId < 1000:
                    semanticId = segmentId
                else:
                    semanticId = segmentId // 1000

                labelInfo = id2label[semanticId]
                categoryId = labelInfo.trainId if useTrainId else labelInfo.id
                if labelInfo.ignoreInEval:
                    continue

                if labelInfo.name not in depth_map.keys():
                    depth_map[labelInfo.name] = [[] for i in range(0, len(bins))]

                # find the object center and compute its area and add it to the corresponding list
                mask = originalFormat == segmentId
                area = np.sum(mask)

                x_min = np.min(np.where(mask == True)[0])
                x_max = np.max(np.where(mask == True)[0])

                y_min = np.min(np.where(mask == True)[1])
                y_max = np.max(np.where(mask == True)[1])

                x_center = x_min + int((x_max - x_min) / 2)
                y_center = y_min + int((y_max - y_min) / 2)
                bin_at_center = round(np.mean(digitized[x_center:x_center+10:, y_center:y_center+10]))
                depth_map[labelInfo.name][bin_at_center].append(area)

            print("\rProgress: {:>3.2f} %".format((i + 1) * 100 / 1000), end=' ')
            if i == 1000:
                break

        scale_map = {}
        for key, depth_bins in depth_map.items():
            scale_map[key] = []
            for bin in depth_bins:
                if len(bin) > 0:
                    mean = np.mean(np.array(bin))
                else:
                    mean = 0
                scale_map[key].append(int(mean))

        scaleOutFile = os.path.join(outputFolder, parent_folder, scale_output_json)
        print("\nSaving the scale json file {}".format(scaleOutFile))

        with open(scaleOutFile, 'w') as f:
            json.dump(scale_map, f, sort_keys=True, indent=4)

    open(os.path.join(outputFolder, parent_folder, status_file_name), 'w').close()

    images = []
    annotations = []
    instance_count = 0

    for progress, f in enumerate(bdd_files):
        random.seed(instance_count)
        originalFormat = np.array(Image.open(f))

        fileName = os.path.basename(f)

        print("\rProgress: {:>3.2f} % , {}".format((progress + 1) * 100 / len(bdd_files), fileName), end=' ')
        # to monitor the progress from remote
        with open(os.path.join(outputFolder, parent_folder, status_file_name), 'a') as f:
            f.write("\nProgress: {:>3.2f} % , {}. web image ID's used -> ".format(
                (progress + 1) * 100 / len(bdd_files), fileName))

        bdd_imageId = fileName.replace("_gtFine_instanceIds.png", "")
        bdd_inputFileName = fileName.replace("_gtFine_instanceIds.png", "_leftImg8bit.png")
        bdd_labelIdsName = fileName.replace("_instanceIds.png", "_labelIds.png")
        bdd_depth_name = fileName.replace("_gtFine_instanceIds.png", "_depth.png")
        city_name = bdd_inputFileName.split("_")[0]
        bdd_img_path = os.path.join(bddPath, "leftImg8bit", bddSplit, city_name,
                                           bdd_inputFileName)
        bdd_labelIDs_path = os.path.join(bddPath, "gtFine", bddSplit, city_name,
                                                bdd_labelIdsName)
        bdd_depth_path = os.path.join(bddPath, "depth", bddSplit, city_name,
                                                 bdd_depth_name)
        bdd_depth_img = np.array(Image.open(bdd_depth_path))

        outputFileName = fileName.replace("_instanceIds.png", "_panoptic.png")
        # image entry, id for image is its filename without extension
        images.append({"bdd_id": bdd_imageId,
                       "width": int(originalFormat.shape[1]),
                       "height": int(originalFormat.shape[0]),
                       "file_name": bdd_inputFileName})
        bdd_leftImg8bit_img = np.array(Image.open(bdd_img_path))
        bdd_labelIds_img = np.array(Image.open(bdd_labelIDs_path))

        bdd_pan_format = np.zeros(
            (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
        )

        segmentIds = np.unique(originalFormat)
        segmInfo = []
        for segmentId in segmentIds:
            if segmentId < 1000:
                semanticId = segmentId
                isCrowd = 1
            else:
                semanticId = segmentId // 1000
                isCrowd = 0
            labelInfo = id2label[semanticId]
            categoryId = labelInfo.trainId if useTrainId else labelInfo.id
            if labelInfo.ignoreInEval:
                continue
            if not labelInfo.hasInstances:
                isCrowd = 0

            mask = originalFormat == segmentId
            color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
            bdd_pan_format[mask] = color

            area = np.sum(mask)  # segment area computation

            # bbox computation for a segment
            hor = np.sum(mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]

            segmInfo.append({"id": int(segmentId),
                             "category_id": int(categoryId),
                             "area": int(area),
                             "bbox": bbox,
                             "iscrowd": isCrowd})

        # add ood objects from web to bdd image
        # sample many random images for each bdd image.

        sample_instance_annotations = []
        random_instances = random.randint(1, random_no_instances)
        for sample_instance_count in range(sample_instances):
            random.seed((instance_count + 1) * 100000 + sample_instance_count)
            # print("random_seed " , (instance_count+1)*100000 + sample_instance_count)

            bdd_labelIDs = np.copy(bdd_labelIds_img)
            bdd_instanceIDs = np.copy(originalFormat)

            sample_segment_info = []
            sample_ood_panoptic_name = "{}_gtFine_panoptic_{}.png".format(bdd_imageId,
                                                                          str(sample_instance_count))
            sample_ood_image_name = "{}_leftImg8bit_{}.png".format(bdd_imageId, str(sample_instance_count))
            sample_fileName = "{}_gtFine_instanceIds_{}.png".format(bdd_imageId, str(sample_instance_count))
            sample_bdd_labelIdsName = "{}_gtFine_labelIds_{}.png".format(bdd_imageId,
                                                                                str(sample_instance_count))
            web_masked_image = np.zeros((city_height, city_width, 3), dtype="uint8")
            bdd_masked_image = np.ones((city_height, city_width, 3), dtype="uint8")

            web_masked_panoptic = np.zeros((city_height, city_width, 3), dtype="uint8")
            bdd_masked_panoptic = np.ones((city_height, city_width, 3), dtype="uint8")
            bdd_web_panoptic_values = np.zeros((city_height, city_width), dtype=np.int32)

            roadPixelID = 0 if useTrainId else 7
            road_pixel_mask = np.zeros((city_height, city_width), dtype="uint8")
            road_pixels = np.where(originalFormat == roadPixelID)
            road_pixel_mask[road_pixels] = 1


            skyPixelID = 10 if useTrainId else 23
            sky_pixel_mask = np.zeros((city_height, city_width), dtype="uint8")
            # indicates the objects can be anywhere
            sky_pixels = np.where(originalFormat >= 0)
            sky_pixel_mask[sky_pixels] = 1

            # thic counts individual instances in each image
            random_instance_count = 0

            # counts individual web instance images
            web_image_instance_count = 0

            for i in range(random_instances):
                # 1370
                id = web_img_Ids[(instance_count + web_image_instance_count) % len(web_img_Ids)]

                # write content to status file
                if sample_instance_count == 0:
                    print("web_image_count = ", (instance_count + web_image_instance_count))
                    with open(os.path.join(outputFolder, parent_folder, status_file_name), 'a') as f:
                        f.write("{}, ".format(id))

                instance = web_json_data[id]
                bbox = instance["bbox"]
                segment_name = instance["semantic_names"][0].strip()

                web_panoptic = np.asarray(Image.open(os.path.join(webInstancePath, instance["panoptic_image"])))
                web_image = np.asarray(Image.open(os.path.join(webInstancePath, instance["rgb_image"])))
                web_semantic = np.asarray(Image.open(os.path.join(webInstancePath, instance["semantic_image"])))

                web_panoptic = web_panoptic[bbox[0]: bbox[2], bbox[1]: bbox[3]]
                web_image = web_image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
                web_semantic = web_semantic[bbox[0]: bbox[2], bbox[1]: bbox[3]]
                web_h, web_w = web_panoptic.shape


                end_width = city_width - web_w

                location_type = web_categories[segment_name][0]

                road_pixels = np.where(road_pixel_mask > 0)
                # in some images there are no road pixels in GT.
                # in such cases consider any pixel
                if len(road_pixels[0]) == 0:
                    road_pixels = np.where(sky_pixel_mask > 0)
                sky_pixels = np.where(sky_pixel_mask > 0)


                road_random_pixel = random.randint(0, len(road_pixels[0]))
                start_width = road_pixels[1][road_random_pixel]
                start_height = road_pixels[0][road_random_pixel] - web_h

                # in some image road pixels have wrong GT marking. To overcome that we use this loop
                '''while not (start_width > 0 and start_height > 0):
                    road_random_pixel = random.randint(0, len(road_pixels[0]))
                    start_width = road_pixels[1][road_random_pixel]
                    start_height = road_pixels[0][road_random_pixel] - web_h'''

                # incase of grouped images scaling based on first object will scale all other objects
                cityscape_relative_instance = web_categories[segment_name][1]
                cityscape_relative_scale_factor = web_categories[segment_name][2]

                #depth_at_start_pixel = bdd_depth_img[start_height][start_width]
                # consider averge of sorrounding pixels for more efficient
                average_depth = np.mean(
                    bdd_depth_img[road_pixels[0][road_random_pixel]:road_pixels[0][road_random_pixel] + web_h, start_width: start_width + web_w])
                depth_at_start_pixel = average_depth
                bin = np.digitize(np.array(depth_at_start_pixel), bins)

                # exception cases
                if bin == 0:
                    # assign it to some non zero bin
                    bin = 2
                if bin >= len(bins):
                    bin = len(bins) - 1

                average_area = int(scale_map[cityscape_relative_instance][int(bin)] * cityscape_relative_scale_factor)

                # segment_name, cityscape_relative_instance, depth_at_start_pixel, average_depth, bin, scale_map[cityscape_relative_instance], (start_height , start_width), bdd_imageId)

                # if the scale map value is zero then it means we dont have an example in training data with corresponding depth value
                # in that case get the next smallest bin size of the object
                if average_area == 0:
                    non_zero_bins = np.where(
                        np.where(np.array(scale_map[cityscape_relative_instance]) > 0)[0] < int(bin))
                    if len(non_zero_bins) > 0:
                        average_area = scale_map[cityscape_relative_instance][
                                           non_zero_bins[0][-1]] * cityscape_relative_scale_factor

                mask = web_panoptic == instance["instance_ids"][0]
                # segment area computation
                area = np.sum(mask)

                scaling_factor = math.sqrt(average_area / area)
                scaled_h = int(web_h * scaling_factor)
                scaled_w = int(web_w * scaling_factor)

                web_panoptic = cv2.resize(web_panoptic, dsize=(scaled_w, scaled_h),
                                           interpolation=cv2.INTER_NEAREST)
                web_image = cv2.resize(web_image, dsize=(scaled_w, scaled_h),
                                        interpolation=cv2.INTER_NEAREST)

                web_scaled_h, web_scaled_w = web_panoptic.shape

                start_height = road_pixels[0][road_random_pixel] - web_scaled_h

                # remove the part which goes out of boundary after rescaling
                if start_width + web_scaled_w >= city_width:
                    diff = (start_width + web_scaled_w) - city_width
                    web_panoptic = web_panoptic[:, :-diff]
                    web_image = web_image[:, :-diff, :]

                # remove the part which goes out of boundary after rescaling
                if start_height <= 0:
                    web_panoptic = web_panoptic[-start_height:, :]
                    web_image = web_image[-start_height:, :, :]

                web_scaled_h, web_scaled_w = web_panoptic.shape


                '''diff_h = web_h - web_scaled_h
                diff_w = web_w - web_scaled_w

                # since the width and height of scaled image is changes we need to alter the start positions of overlap
                start_width += diff_w
                start_height += diff_h '''

                # print(scaling_factor, (average_area, area), (web_h, web_w), (scaled_h, scaled_w))
                web_h = web_scaled_h
                web_w = web_scaled_w

                start_height = road_pixels[0][road_random_pixel] - web_h

                '''color1 = np.array([0, 255, 0], dtype=np.uint8)
                bdd_masked_image[start_height : start_height +10, start_width: start_width+10] = color1'''

                web_panoptic_format = np.zeros((web_h, web_w, 3), dtype="uint8")
                web_image_format = np.zeros((web_h, web_w, 3), dtype="uint8")
                web_panoptic_values = np.zeros((web_h, web_w), dtype=np.int32)

                for index, web_instance_id in enumerate(instance["instance_ids"]):
                    semantic_id = web_instance_id // 1000

                    segmentId = semantic_id * 1000 + random_instance_count

                    categoryId = np.max(web_semantic)
                    isCrowd = 0

                    mask = web_panoptic == web_instance_id
                    # segment area computation
                    area = np.sum(mask)

                    # bbox computation for a segment
                    hor = np.sum(mask, axis=0)
                    vert = np.sum(mask, axis=1)

                    if len(np.nonzero(hor)[0]) == 0 or len(np.nonzero(vert)[0]) == 0:
                        continue
                    hor_idx = np.nonzero(hor)[0]
                    x = hor_idx[0]
                    width = hor_idx[-1] - x + 1
                    vert_idx = np.nonzero(vert)[0]
                    y = vert_idx[0]
                    height = vert_idx[-1] - y + 1
                    # add start width and height to accommodate the bounding box positions in original bdd size
                    bbox = [int(start_width) + int(x), int(start_height) + int(y), int(width), int(height)]

                    color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                    web_panoptic_format[mask] = color
                    web_image_format[mask] = web_image[mask]
                    web_panoptic_values[mask] = segmentId

                    '''# draw bounding box
                    color = np.array([0, 255, 0], dtype=np.uint8)
                    bdd_masked_image[start_height + int(y): start_height + int(y) + int(height), start_width + int(x): start_width + int(x) + 2] = color
                    bdd_masked_image[start_height + int(y): start_height + int(y) + int(height), start_width + int(x)+int(width): start_width + int(x)+int(width) + 2] = color
                    bdd_masked_image[start_height + int(y): start_height + int(y)+2 , start_width + int(x): start_width + int(x) + int(width)] = color
                    bdd_masked_image[start_height + int(y) + int(height): start_height + int(y)  + int(height)+2, start_width + int(x): start_width + int(x) + int(width)] = color
                    '''

                    sample_segment_info.append({"id": int(segmentId),
                                                "category_id": int(categoryId),
                                                "area": int(area),
                                                "bbox": bbox,
                                                "iscrowd": isCrowd})

                    random_instance_count += 1

                '''# extract only OOD object pixels
                # clear out if already existing ood instances in this location
                web_masked_image[start_height:start_height + web_h, start_width:start_width + web_w, :] = np.zeros(web_image_format.shape)
                web_masked_image[start_height:start_height + web_h, start_width:start_width + web_w, :] += web_image_format


                # same steps for panoptic mask creation
                web_masked_panoptic[start_height:start_height + web_h, start_width:start_width + web_w, :] = np.zeros(web_panoptic_format.shape)
                web_masked_panoptic[start_height:start_height + web_h, start_width:start_width + web_w, :] += web_panoptic_format

                # for saving web panoptic values
                bdd_web_panoptic_values[start_height:start_height + web_h, start_width:start_width + web_w] = np.zeros(web_panoptic_values.shape)
                bdd_web_panoptic_values[start_height:start_height + web_h, start_width:start_width + web_w] += web_panoptic_values
                '''
                # extract only OOD object pixels
                # clear out if already existing ood instances in this location
                i_mask = np.where(web_image_format > 0)
                i_copy_mask = (i_mask[0] + start_height, i_mask[1] + start_width, i_mask[2])
                web_masked_image[i_copy_mask] = 0
                web_masked_image[i_copy_mask] += web_image_format[i_mask]

                # same steps for panoptic mask creation
                p_mask = np.where(web_panoptic_format > 0)
                p_copy_mask = (p_mask[0] + start_height, p_mask[1] + start_width, p_mask[2])
                web_masked_panoptic[p_copy_mask] = 0
                web_masked_panoptic[p_copy_mask] += web_panoptic_format[p_mask]

                # for saving web panoptic values
                v_mask = np.where(web_panoptic_values > 0)
                v_copy_mask = (v_mask[0] + start_height, v_mask[1] + start_width)
                bdd_web_panoptic_values[v_copy_mask] = 0
                bdd_web_panoptic_values[v_copy_mask] += web_panoptic_values[v_mask]

                bdd_labelIDs[v_copy_mask] = ood_id

                # if there are any road pixels in this bounding box then set them to 0
                # which means there are no road pixels inthis location any more
                road_pixel_mask[start_height:start_height + web_h, start_width:start_width + web_w] = 0

                web_image_instance_count += 1

            # make all the ood pixels in bdd image to zero
            # then add OOD pixels into these locations
            mask = np.where(web_masked_image > 0)
            bdd_masked_image[mask] = 0
            bdd_masked_image *= bdd_leftImg8bit_img
            bdd_masked_image += web_masked_image

            mask = np.where(web_masked_panoptic > 0)
            bdd_masked_panoptic[mask] = 0
            bdd_masked_panoptic *= bdd_pan_format
            bdd_masked_panoptic += web_masked_panoptic

            mask = np.where(bdd_web_panoptic_values > 0)
            bdd_instanceIDs[mask] = 0
            bdd_instanceIDs += bdd_web_panoptic_values

            bdd_masked_image = apply_blending(mask, bdd_web_panoptic_values, bdd_depth_img,
                                                     bdd_masked_image, bdd_leftImg8bit_img)

            Image.fromarray(bdd_masked_panoptic).save(os.path.join(outputBaseFile, sample_ood_panoptic_name))

            if not os.path.isdir(os.path.join(left8bitImgPath, city_name)):
                os.makedirs(os.path.join(left8bitImgPath, city_name), exist_ok=True)
            Image.fromarray(bdd_masked_image).save(
                os.path.join(left8bitImgPath, city_name, sample_ood_image_name))

            if not os.path.isdir(os.path.join(labelIdsPath, city_name)):
                os.makedirs(os.path.join(labelIdsPath, city_name), exist_ok=True)

            Image.fromarray(bdd_labelIDs).save(
                os.path.join(labelIdsPath, city_name, sample_bdd_labelIdsName))
            Image.fromarray(bdd_instanceIDs).save(os.path.join(labelIdsPath, city_name, sample_fileName))

            sample_instance_annotations.append({'image_id': bdd_imageId,
                                                'file_name': outputFileName,
                                                "segments_info": segmInfo + sample_segment_info})

        annotations.append(sample_instance_annotations)

        instance_count += web_image_instance_count

        sys.stdout.flush()

    print("\nSaving the json file {}".format(outFile))
    d = {'images': images,
         'annotations': annotations,
         'categories': categories}
    with open(outFile, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd-folder",
                        dest="bddPath",
                        help="path to the Bdd dataset 'gtFine' folder",
                        default="/home/kumarasw/OOD_dataset/bdd/bdd",
                        type=str)
    parser.add_argument("--bdd-split",
                        dest="bddSplit",
                        help="bdd data split to be used to create the OOD dataset",
                        default="val",
                        type=str)
    parser.add_argument("--web-instance-path",
                        dest="webInstancePath",
                        help="path to the COCO dataset folder",
                        default="/home/kumarasw/OOD_dataset/web_ood/filtered_ood_instances",
                        type=str)
    parser.add_argument("--ood-split",
                        dest="oodSplit",
                        help="web data split to be used to create the OOD dataset",
                        default="test",
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default="/home/kumarasw/OOD_dataset/bdd/",
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")

    args = parser.parse_args()

    create_bdd_web_panoptic(args)


# call the main
if __name__ == "__main__":
    main()