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

# Image processing
from PIL import Image

# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.helpers.labels import id2label, labels, trainId2label
from pycocotools.coco import COCO as coco_tools
import random
from extract_coco_instances import coco_categories

ood_train_id = 20
ood_id = 50

city_height = 1024
city_width = 2048
bins = np.linspace(0, 40000, 20)

def apply_blending(mask, cityscapes_coco_panoptic_values, cityscapes_disparity_img, cityscapes_masked_image, cityscapes_original):
    #apply smoothing of ood object at border

    #Load original image and find contours.
    #Blur the original image and save it in a different variable.
    #Create an empty mask and draw the detected contours on it.
    #Use np.where() method to select the pixels from the mask (contours) where you want blurred values and then replace it.

    smooth_mask = np.zeros((city_height, city_width), dtype="uint8")
    smooth_mask[mask[0:2]] = 255
    Image.fromarray(cityscapes_masked_image).save(os.path.join("/home/kumarasw/kiran/cityscapes_coco", "no_blending.png"))


    blurred_img = cv2.blur(cityscapes_masked_image, (5, 5))
    contour_mask = np.zeros(blurred_img.shape, np.uint8)

    contours, hierarchy = cv2.findContours(image=smooth_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_copy = np.copy(cityscapes_masked_image)
    cv2.drawContours(image=contour_mask, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2,
                     lineType=cv2.LINE_AA)

    cityscapes_masked_image = np.where(contour_mask == np.array([255, 255, 255]), blurred_img, cityscapes_masked_image)

    Image.fromarray(cityscapes_masked_image).save(os.path.join("/home/kumarasw/kiran/cityscapes_coco", "smoothed.png"))
    Image.fromarray(contour_mask).save(os.path.join("/home/kumarasw/kiran/cityscapes_coco", "contours.png"))

    # Apply brighness to OOD pixels to match cityscapes pixels
    hsv_city = cv2.cvtColor(cityscapes_original, cv2.COLOR_BGR2HSV)
    V_city_mask = hsv_city[:,:,2][np.where(smooth_mask==255)]
    avg_V_city_mask = np.mean(V_city_mask)

    hsv_coco = cv2.cvtColor(cityscapes_masked_image, cv2.COLOR_BGR2HSV)
    V_coco_mask = hsv_coco[:, :, 2][np.where(smooth_mask == 255)]
    avg_V_coco_mask = np.mean(V_coco_mask)

    diff_V = avg_V_city_mask - avg_V_coco_mask

    hsv_new = hsv_coco[:, :, 2][np.where(smooth_mask == 255)] + int(diff_V)
    hsv_new[np.where(hsv_new < 0)] = 0
    hsv_new[np.where(hsv_new > 255)] = 255

    hsv_coco[:, :, 2][np.where(smooth_mask == 255)] = hsv_new

    cityscapes_masked_image = cv2.cvtColor(hsv_coco, cv2.COLOR_HSV2BGR)


    Image.fromarray(cityscapes_masked_image).save(os.path.join("/home/kumarasw/kiran/cityscapes_coco", "brightness.png"))

    # apply depth blur at OOD. We use disparity values to vary kernel size of Gaussianblur .
    panoptic_ids = np.unique(cityscapes_coco_panoptic_values).tolist()
    panoptic_ids.remove(0)

    for id in panoptic_ids:
        mask = np.where(cityscapes_coco_panoptic_values==id)
        avg_disparity = cityscapes_disparity_img[mask].mean()
        bin = np.digitize(avg_disparity, bins)
        kernel_size = int((len(bins) - bin)/3)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size - 1
        blurred_img = cv2.GaussianBlur(cityscapes_masked_image, (kernel_size, kernel_size), 0)
        print("Kernel_size ",  kernel_size)
        cityscapes_masked_image[mask] = blurred_img[mask]

    Image.fromarray(cityscapes_masked_image).save(os.path.join("/home/kumarasw/kiran/cityscapes_coco", "gaussian_depth_blur.png"))

    # adding color noise
    noise = np.random.randint(0, 4, cityscapes_masked_image.shape)
    noisy_cityscapes = cityscapes_masked_image + noise
    noisy_cityscapes[np.where(noisy_cityscapes > 255)] = 255
    cityscapes_masked_image = noisy_cityscapes.astype(np.uint8)
    Image.fromarray(cityscapes_masked_image).save(os.path.join("/home/kumarasw/kiran/cityscapes_coco", "noise.png"))

    return cityscapes_masked_image


# The main method
def create_cityscapes_coco_panoptic(cityscapesPath=None, cityscapesSplit=None, cocoPath=None, cocoSplit=None, cocoYear=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    # Where to look for Cityscapes
    if cityscapesPath is None:
        if 'CITYSCAPES_DATASET' in os.environ:
            cityscapesPath = os.environ['CITYSCAPES_DATASET']
        cityscapesPath = os.path.join(cityscapesPath, "gtFine")

    if cityscapesPath is None:
        if 'COCO_DATASET' in os.environ:
            cocoPath = os.environ['COCO_DATASET']

    if outputFolder is None:
        raise "Please specify output directory."

    coco_instances_dir = os.path.join(outputFolder, "coco_instances")
    coco_images_folder = "coco_rgb"
    coco_panoptic_folder = "coco_panoptic"
    coco_semantic_folder = "coco_semantic"

    coco_img_Ids = []

    coco_json_file = os.path.join(outputFolder, "coco_instances.json")
    coco_json_data = json.load(open(coco_json_file))
    for coco_id, value in coco_json_data.items():
        if (os.path.exists(coco_json_data[coco_id]["rgb_image"]) and
            os.path.exists(coco_json_data[coco_id]["panoptic_image"]) and
            os.path.exists(coco_json_data[coco_id]["semantic_image"])):
            coco_img_Ids.append(coco_id)
        else:
            print("missing ", coco_id)

    mode = "gtFine"

    categories = []
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                           'name': label.name,
                           'color': label.color,
                           'supercategory': label.category,
                           'isthing': 1 if label.hasInstances else 0})

    searchFine = os.path.join(cityscapesPath, mode, cityscapesSplit, "*", "*_instanceIds.png")
    # search files
    filesFine = glob.glob(searchFine)
    filesFine.sort()

    cityscapes_files = filesFine

    # quit if we did not find anything
    if not cityscapes_files:
        printError(
            "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(
                cityscapesSplit, searchFine)
        )
    # a bit verbose
    print("Converting {} annotation files for {} set.".format(len(cityscapes_files), cityscapesSplit))


    outputBaseFile = os.path.join(outputFolder, "cityscapes_"+cityscapesSplit+"_coco_"+cocoSplit)
    outFile = outputBaseFile+".json"

    if not os.path.isdir(outputBaseFile):
        print("Creating folder {} for saving OOD images and ground truth".format(outputBaseFile))
        os.makedirs(outputBaseFile, exist_ok=True)
    print("Corresponding segmentation ground truth will be saved in {}".format(outFile))

    # load instance scale json file if exists else create a new one
    # This file contains the average size of the cityscapes instance at certain distance divided into bins
    scale_output_json = "instance_scale.json"    

    if os.path.exists(os.path.join(outputFolder, scale_output_json)):
        scale_map = json.load(open(os.path.join(outputFolder, scale_output_json)))

    else:
        print("Creating mapping for image size and disparity. Please wait as it might take longer time.")
        # map to store the average object size for each class and at given depth(no of bins)
        disparity_map = {}
        print("Total images: ", len(cityscapes_files))
        for i, f in enumerate(cityscapes_files):
            print(i)
            originalFormat = np.array(Image.open(f))
            fileName = os.path.basename(f)
            cityscapes_imageId = fileName.replace("_gtFine_instanceIds.png", "")
            cityscapes_disparity_name = fileName.replace("_gtFine_instanceIds.png", "_disparity.png")
            city_name = cityscapes_disparity_name.split("_")[0]
            cityscapes_disparity_path = os.path.join(cityscapesPath, "disparity", cityscapesSplit, city_name, cityscapes_disparity_name)
            cityscapes_disparity_img = np.array(Image.open(cityscapes_disparity_path))

            Image.fromarray(cityscapes_disparity_img).save(os.path.join(outputFolder, "disparity.png"))
            Image.fromarray(originalFormat).save(os.path.join(outputFolder, "panoptic.png"))


            digitized = np.digitize(cityscapes_disparity_img, bins)

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

                if labelInfo.name not in disparity_map.keys():
                    disparity_map[labelInfo.name] = [[]for i in range(0, len(bins))]

                # find the object center and compute its area and add it to the corresponding list
                mask = originalFormat == segmentId
                area = np.sum(mask)

                x_min = np.min(np.where(mask == True)[0])
                x_max = np.max(np.where(mask == True)[0])

                y_min = np.min(np.where(mask == True)[1])
                y_max = np.max(np.where(mask == True)[1])

                x_center = x_min + int((x_max - x_min)/2)
                y_center = y_min + int((y_max - y_min)/2)

                bin_at_center = digitized[x_center][y_center]
                disparity_map[labelInfo.name][bin_at_center].append(area)

            '''if i == 200:
                break'''

        scale_map = {}
        for key, disparity_bins in disparity_map.items():
            scale_map[key] = []
            for bin in disparity_bins:
                if len(bin) > 0:
                    mean = np.mean(np.array(bin))
                else:
                    mean = 0
                scale_map[key].append(int(mean))


        outFile = os.path.join(outputFolder, scale_output_json)
        print("\nSaving the scale json file {}".format(outFile))

        with open(outFile, 'w') as f:
            json.dump(scale_map, f, sort_keys=True, indent=4)


    images = []
    annotations = []
    instance_count = 0
    for progress, f in enumerate(cityscapes_files):
        originalFormat = np.array(Image.open(f))

        fileName = os.path.basename(f)
        cityscapes_imageId = fileName.replace("_gtFine_instanceIds.png", "")
        cityscapes_inputFileName = fileName.replace("_gtFine_instanceIds.png", "_leftImg8bit.png")
        cityscapes_disparity_name = fileName.replace("_gtFine_instanceIds.png", "_disparity.png")
        city_name = cityscapes_inputFileName.split("_")[0]
        cityscapes_img_path = os.path.join(cityscapesPath, "leftImg8bit", cityscapesSplit, city_name, cityscapes_inputFileName)
        cityscapes_disparity_path = os.path.join(cityscapesPath, "disparity", cityscapesSplit, city_name,
                                                 cityscapes_disparity_name)
        cityscapes_disparity_img = np.array(Image.open(cityscapes_disparity_path))


        
        outputFileName = fileName.replace("_instanceIds.png", "_panoptic.png")
        # image entry, id for image is its filename without extension
        images.append({"cityscapes_id": cityscapes_imageId,
                       "width": int(originalFormat.shape[1]),
                       "height": int(originalFormat.shape[0]),
                       "file_name": cityscapes_inputFileName})
        cityscapes_leftImg8bit_img = np.array(Image.open(cityscapes_img_path))

        cityscapes_pan_format = np.zeros(
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
            cityscapes_pan_format[mask] = color

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

        # add ood objects from coco to cityscapes image

        segment_info = []
        ood_panoptic_name = "{}_panoptic.png".format(cityscapes_imageId)
        ood_image_name = "{}_leftImg8bit.png".format(cityscapes_imageId)

        coco_masked_image = np.zeros((city_height, city_width, 3), dtype="uint8")
        cityscapes_masked_image = np.ones((city_height, city_width, 3), dtype="uint8")

        coco_masked_panoptic = np.zeros((city_height, city_width, 3), dtype="uint8")
        cityscapes_masked_panoptic = np.ones((city_height, city_width, 3), dtype="uint8")
        cityscapes_coco_panoptic_values = np.zeros((city_height, city_width))

        roadPixelID = 0 if useTrainId else 7
        road_pixels = np.where(originalFormat == roadPixelID)

        random_instances = random.randint(1, 2)
        random_instance_count = 1
        for i in range (random_instances):
            id = coco_img_Ids[instance_count % len(coco_img_Ids)]
            instance = coco_json_data[id]
            bbox = instance["bbox"]

            coco_panoptic = np.asarray(Image.open(instance["panoptic_image"]))
            coco_image = np.asarray(Image.open(instance["rgb_image"]))
            coco_semantic = np.asarray(Image.open(instance["semantic_image"]))


            coco_panoptic = coco_panoptic[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            coco_image = coco_image[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            coco_semantic = coco_semantic[bbox[0]: bbox[2], bbox[1]: bbox[3]]
            coco_h, coco_w = coco_panoptic.shape


            end_width = city_width - coco_w

            road_end_h = np.where(road_pixels[0] > coco_h)
            road_end_w = np.where(road_pixels[1] < end_width)

            road_eligible_pixels = np.intersect1d(road_end_h, road_end_w)

            road_random_pixel = random.randint(0, len(road_eligible_pixels))

            start_width = road_pixels[1][road_eligible_pixels[road_random_pixel]]
            start_height = road_pixels[0][road_eligible_pixels[road_random_pixel]] - coco_h

            color1 = np.array([255, 0, 0], dtype=np.uint8)
            cityscapes_masked_image[road_pixels[0][road_eligible_pixels[road_random_pixel]] : road_pixels[0][road_eligible_pixels[road_random_pixel]] +10, start_width: start_width+10] = color1
            

            # incase of grouped images scaling based on first object will scale all other objects
            segment_name = instance["semantic_names"][0]
            cityscape_relative_instance = coco_categories[segment_name][1]
            cityscape_relative_scale_factor = coco_categories[segment_name][2]

            disparity_at_start_pixel = cityscapes_disparity_img[start_height][start_width]
            # consider averge of sorrounding pixels for more efficient
            average_disparity = np.mean(cityscapes_disparity_img[start_height:start_height+coco_h, start_width: start_width+coco_w])
            disparity_at_start_pixel = average_disparity
            bin = np.digitize(np.array(disparity_at_start_pixel), bins)

            average_area = int(scale_map[cityscape_relative_instance][int(bin)] * cityscape_relative_scale_factor)           
            
            
            print(segment_name, cityscape_relative_instance, disparity_at_start_pixel, average_disparity, bin, scale_map[cityscape_relative_instance], (start_height , start_width), cityscapes_imageId)

            # if the scale map value is zero then it means we dont have an example in training data with corresponding disparity value
            # in that case get the next smallest bin size of the object
            if average_area == 0:
                non_zero_bins = np.where(np.where(np.array(scale_map[cityscape_relative_instance]) > 0)[0] <int(bin))
                if len(non_zero_bins) > 0:
                    average_area = scale_map[cityscape_relative_instance][non_zero_bins[-1]] * cityscape_relative_scale_factor

            mask = coco_panoptic == instance["instance_ids"][0]
            # segment area computation
            area = np.sum(mask)


            scaling_factor = math.sqrt(average_area / area)
            scaled_h = int(coco_h * scaling_factor)
            scaled_w = int(coco_w * scaling_factor)            

            
            coco_panoptic= cv2.resize(coco_panoptic, dsize=(scaled_w, scaled_h),
                                              interpolation=cv2.INTER_NEAREST)
            coco_image = cv2.resize(coco_image, dsize=(scaled_w, scaled_h),
                                           interpolation=cv2.INTER_NEAREST)

            coco_scaled_h, coco_scaled_w = coco_panoptic.shape

            start_height = road_pixels[0][road_eligible_pixels[road_random_pixel]] - coco_scaled_h

            # remove the part which goes out of boundary after rescaling
            if start_width + coco_scaled_w >= city_width:
                diff = (start_width + coco_scaled_w) - city_width
                coco_panoptic = coco_panoptic[:, :-diff]
                coco_image = coco_image[:, :-diff, :]

            # remove the part which goes out of boundary after rescaling
            if start_height <= 0:                
                coco_panoptic = coco_panoptic[-start_height:, :]
                coco_image = coco_image[-start_height:, :, :]

            
            coco_scaled_h, coco_scaled_w = coco_panoptic.shape

            '''diff_h = coco_h - coco_scaled_h
            diff_w = coco_w - coco_scaled_w

            # since the width and height of scaled image is changes we need to alter the start positions of overlap
            start_width += diff_w
            start_height += diff_h '''
            
            print(scaling_factor, (average_area, area), (coco_h, coco_w), (scaled_h, scaled_w))
            coco_h = coco_scaled_h
            coco_w = coco_scaled_w

            start_height = road_pixels[0][road_eligible_pixels[road_random_pixel]] - coco_h

            '''color1 = np.array([0, 255, 0], dtype=np.uint8)
            cityscapes_masked_image[start_height : start_height +10, start_width: start_width+10] = color1'''

                        

            coco_panoptic_format = np.zeros((coco_h, coco_w, 3), dtype="uint8")
            coco_image_format = np.zeros((coco_h, coco_w, 3), dtype="uint8")
            coco_panoptic_values = np.zeros((coco_h, coco_w))

            for index, coco_instance_id in enumerate(instance["instance_ids"]):
                semantic_id = coco_instance_id // 1000

                segmentId = semantic_id * 1000 + random_instance_count

                categoryId = np.max(coco_semantic)
                isCrowd = 0

                mask = coco_panoptic == coco_instance_id
                # segment area computation
                area = np.sum(mask)

                color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                coco_panoptic_format[mask] = color
                coco_image_format[mask] = coco_image[mask]
                coco_panoptic_values[mask] = segmentId

                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                # add start width and height to accommodate the bounding box positions in original cityscapes size
                bbox = [start_width + int(x), start_height+ int(y), int(width), int(height)]

                # draw bounding box
                color = np.array([0, 255, 0], dtype=np.uint8)
                cityscapes_masked_image[start_height + int(y): start_height + int(y) + int(height), start_width + int(x): start_width + int(x) + 2] = color
                cityscapes_masked_image[start_height + int(y): start_height + int(y) + int(height), start_width + int(x)+int(width): start_width + int(x)+int(width) + 2] = color
                cityscapes_masked_image[start_height + int(y): start_height + int(y)+2 , start_width + int(x): start_width + int(x) + int(width)] = color
                cityscapes_masked_image[start_height + int(y) + int(height): start_height + int(y)  + int(height)+2, start_width + int(x): start_width + int(x) + int(width)] = color


                segmInfo.append({"id": int(segmentId),
                                 "category_id": int(categoryId),
                                 "area": int(area),
                                 "bbox": bbox,
                                 "iscrowd": isCrowd})

                random_instance_count += 1


            # extract only OOD object pixels
            # clear out if already existing ood instances in this location
            coco_masked_image[start_height:start_height + coco_h, start_width:start_width + coco_w, :] = np.zeros(coco_image_format.shape)
            coco_masked_image[start_height:start_height + coco_h, start_width:start_width + coco_w, :] += coco_image_format


            # same steps for panoptic mask creation
            coco_masked_panoptic[start_height:start_height + coco_h, start_width:start_width + coco_w, :] = np.zeros(coco_panoptic_format.shape)
            coco_masked_panoptic[start_height:start_height + coco_h, start_width:start_width + coco_w, :] += coco_panoptic_format

            # for saving coco panoptic values
            cityscapes_coco_panoptic_values[start_height:start_height + coco_h, start_width:start_width + coco_w] = np.zeros(coco_panoptic_values.shape)
            cityscapes_coco_panoptic_values[start_height:start_height + coco_h, start_width:start_width + coco_w] += coco_panoptic_values


            instance_count += 1


        # make all the ood pixels in cityscapes image to zero
        # then add OOD pixels into these locations
        mask = np.where(coco_masked_image > 0)
        cityscapes_masked_image[mask] = 0
        cityscapes_masked_image *= cityscapes_leftImg8bit_img
        cityscapes_masked_image += coco_masked_image

        mask = np.where(coco_masked_panoptic > 0)
        cityscapes_masked_panoptic[mask] = 0
        cityscapes_masked_panoptic *= cityscapes_pan_format
        cityscapes_masked_panoptic += coco_masked_panoptic

        cityscapes_masked_image = apply_blending(mask, cityscapes_coco_panoptic_values, cityscapes_disparity_img, cityscapes_masked_image, cityscapes_leftImg8bit_img)

        Image.fromarray(cityscapes_masked_panoptic).save(os.path.join(outputBaseFile, ood_panoptic_name))
        Image.fromarray(cityscapes_masked_image).save(os.path.join(outputBaseFile, ood_image_name))

        annotations.append({'image_id': cityscapes_imageId,
                            'file_name': outputFileName,
                            "segments_info": segmInfo})

        print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(cityscapes_files)), end=' ')
        sys.stdout.flush()

    print("\nSaving the json file {}".format(outFile))
    d = {'images': images,
         'annotations': annotations,
         'categories': categories}
    with open(outFile, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes-folder",
                        dest="cityscapesPath",
                        help="path to the Cityscapes dataset 'gtFine' folder",
                        default="/export/kiran/cityscapes",
                        type=str)
    parser.add_argument("--cityscapes-split",
                        dest="cityscapesSplit",
                        help="cityscapes data split to be used to create the OOD dataset",
                        default="val",
                        type=str)
    parser.add_argument("--coco-folder",
                        dest="cocoPath",
                        help="path to the COCO dataset folder",
                        default="/export/kiran/coco/2017",
                        type=str)
    parser.add_argument("--coco-split",
                        dest="cocoSplit",
                        help="coco data split to be used to create the OOD dataset",
                        default="val",
                        type=str)
    parser.add_argument("--coco-year",
                        dest="cocoYear",
                        help="coco dataset year to be used to create the OOD dataset",
                        default="2017",
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default="/home/kumarasw/kiran/cityscapes_coco",
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["val", "train", "test"],
                        type=str)

    args = parser.parse_args()

    create_cityscapes_coco_panoptic(args.cityscapesPath, args.cityscapesSplit, args.cocoPath, args.cocoSplit, args.cocoYear, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()