#!/usr/bin/python
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image

# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.helpers.labels import id2label, labels
from pycocotools.coco import COCO as coco_tools
import random

coco_categories = [
    "airplane",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "skateboard",
    "suitcase",
    "bottle",
    "chair",
    "couch",
    "teddy bear"
]

ood_train_id = 20
ood_id = 50

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

    coco_images_dir = '{}/{}'.format(cocoPath, cocoSplit + str(cocoYear))
    annotation_file = '{}/annotations/instances_{}.json'.format(cocoPath, cocoSplit + str(cocoYear))
    coco_gt_json_file = os.path.join(cocoPath, "annotations", "panoptic_" + cocoSplit + cocoYear + ".json")
    json_data = json.load(open(coco_gt_json_file))
    coco_instances = []

    tools = coco_tools(annotation_file)
    include_cat_Ids = tools.getCatIds(catNms=coco_categories)

    coco_include_img_Ids = []
    for cat_Id in include_cat_Ids:
        coco_include_img_Ids += tools.getImgIds(catIds=cat_Id)
    coco_include_img_Ids = set(coco_include_img_Ids)
    coco_img_Ids = [int(image[:-4]) for image in os.listdir(coco_images_dir) if int(image[:-4]) in coco_include_img_Ids]
    coco_img_Ids.sort()

    print("Total coco images selected: ", len(coco_img_Ids))

    selected_coco_img_Ids = []

    if os.path.exists(os.path.join(outputFolder, "panoptic_mask.npy")):
        instance_panoptic_list = np.load(os.path.join(outputFolder, "panoptic_mask.npy"), allow_pickle=True)
        instance_image_list = np.load(os.path.join(outputFolder, "image_mask.npy"), allow_pickle=True)
        instance_semantic_list = np.load(os.path.join(outputFolder, "semantic_mask.npy"), allow_pickle=True)
    else:

        random_instance_count = random.randint(1, 10)
        total_instances = 0
        instance_count = 1
        random_instances = []
        for i, img_Id in enumerate(coco_img_Ids):
            selected = False
            coco_img = tools.loadImgs(img_Id)[0]
            h, w = coco_img['height'], coco_img['width']
            ann_Ids = tools.getAnnIds(imgIds=coco_img['id'], iscrowd=None)
            annotations = tools.loadAnns(ann_Ids)
            coco_img_id = coco_img["file_name"].split(".")[0]
            image = Image.open(os.path.join(coco_images_dir, coco_img["file_name"])).convert('RGB')
            coco_rgb_img = np.array(image)

            coco_panoptic_mask = np.zeros((h, w), dtype="uint8")
            coco_semantic_mask = np.zeros((h, w), dtype="uint8")


            # Generate binary segmentation mask
            panoptic_mask = np.zeros((h, w), dtype="uint8")

            grouped_instances = {}
            count = 0

            for j in range(len(annotations)):
                category_id = annotations[j]["category_id"]
                bbox = annotations[j]["bbox"]
                # h_min, w_min. h_max, w_max
                bbox = [int(bbox[1]), int(bbox[0]), int(bbox[1])+int(bbox[3]), int(bbox[0])+int(bbox[2])]
                if category_id in include_cat_Ids:
                    categoryId = ood_train_id if useTrainId else ood_id
                    instance_id = categoryId * 1000 + instance_count

                    mask = tools.annToMask(annotations[j])
                    mask_position = np.where(mask > 0)
                    if ((np.max(mask_position[0]) > h - 5) or (np.max(mask_position[1]) > w - 5)
                            or (np.min(mask_position[0]) < 5) or (np.min(mask_position[1]) < 5)):
                        continue
                    area = int(np.sum(mask > 0))

                    if area > 3000:
                        coco_panoptic_mask = mask * instance_id
                        coco_semantic_mask = mask * categoryId
                        coco_image_mask = np.expand_dims(mask, axis=2) * coco_rgb_img

                        if len(grouped_instances.items()) == 0:
                            grouped_instances[count] = {
                                "bbox": bbox,
                                "panoptic_mask": coco_panoptic_mask,
                                "semantic_mask": coco_semantic_mask,
                                "image_mask": coco_image_mask,
                                "instance_ids": [instance_id],
                                "semantic_ids": [categoryId]
                            }
                            count +=1
                            instance_count += 1
                        else:
                            overlap = False
                            for key, value in grouped_instances.items():
                                bbx_a = value["bbox"]
                                p_mask = value["panoptic_mask"]
                                s_mask = value["semantic_mask"]
                                i_mask = value["image_mask"]
                                instance_ids = value["instance_ids"]
                                semantic_ids =  value["semantic_ids"]
                                box_a_mask = np.zeros((h, w), dtype="uint8")
                                box_b_mask = np.zeros((h, w), dtype="uint8")
                                box_a_mask[int(bbx_a[0]):int(bbx_a[2]), int(bbx_a[1]):int(bbx_a[3])] = 1
                                box_b_mask[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])] = 1
                                box_a = np.where(box_a_mask > 0)
                                box_b = np.where(box_b_mask > 0)
                                x_common = np.intersect1d(box_a[0], box_b[0])
                                y_common = np.intersect1d(box_a[1], box_b[1])
                                # check if the two bounding box intersects are not
                                if len(x_common) > 0 and len(y_common) > 0:
                                    overlap = True
                                    p_mask = np.maximum(coco_panoptic_mask,
                                                                    p_mask)
                                    s_mask = np.maximum(coco_semantic_mask,
                                                                    s_mask)
                                    combined_image_mask = np.zeros((h, w), dtype="uint8")
                                    combined_mask_pos = np.where(p_mask > 0)
                                    combined_image_mask[combined_mask_pos] = 1
                                    coco_image_mask = np.expand_dims(combined_image_mask, axis=2) * coco_rgb_img
                                    instance_ids.append(instance_id)
                                    semantic_ids.append(categoryId)

                                    # object bounding box
                                    min_h = np.min(combined_mask_pos[0])
                                    min_w = np.min(combined_mask_pos[1])
                                    max_h = np.max(combined_mask_pos[0])
                                    max_w = np.max(combined_mask_pos[1])
                                    bbox = [min_h, min_w, max_h, max_w]

                                    grouped_instances[key] = {
                                        "bbox": bbox,
                                        "panoptic_mask": p_mask,
                                        "semantic_mask": s_mask,
                                        "image_mask": coco_image_mask,
                                        "instance_ids": instance_ids,
                                        "semantic_ids": semantic_ids
                                    }
                                    instance_count += 1

                            if not overlap:
                                grouped_instances[count] = {
                                    "bbox": bbox,
                                    "panoptic_mask": coco_panoptic_mask,
                                    "semantic_mask": coco_semantic_mask,
                                    "image_mask": coco_image_mask,
                                    "instance_ids": [instance_id],
                                    "semantic_ids": [categoryId]
                                }
                                count += 1
                                instance_count += 1

            for key, value in grouped_instances.items():
                random_instances.append(value)

            if instance_count > random_instance_count:
                total_instances += instance_count
                instance_count = 1
                random_instance_count = random.randint(1, 5)
                coco_instances.append(random_instances)
                random_instances = []

            if total_instances >= 50:
                break



        '''#instance_panoptic_list = np.array(instance_panoptic_list)
        #instance_image_list = np.array(instance_image_list)
        #instance_semantic_list = np.array(instance_semantic_list)
        print("COCO instances extracted: ", len(instance_panoptic_list))
        np.save(os.path.join(outputFolder, "panoptic_mask.npy"), instance_panoptic_list)
        np.save(os.path.join(outputFolder, "image_mask.npy"), instance_image_list)
        np.save(os.path.join(outputFolder, "semantic_mask.npy"), instance_semantic_list)'''


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

    # make cityscapes and coco image length same by duplicating the coco images
    if len(coco_img_Ids) < len(cityscapes_files):
        diff = len(cityscapes_files) - len(coco_img_Ids)
        i = 0
        coco_length = len(coco_img_Ids)
        additional_coco = []
        for count in range(diff):
            additional_coco.append(coco_img_Ids[i])
            i = i+1
            if i == len(coco_img_Ids):
                i = 0
        coco_img_Ids = coco_img_Ids + additional_coco

    outputBaseFile = os.path.join(outputFolder, "cityscapes_"+cityscapesSplit+"_coco_"+cocoSplit)
    outFile = outputBaseFile+".json"

    if not os.path.isdir(outputBaseFile):
        print("Creating folder {} for saving OOD images and ground truth".format(outputBaseFile))
        os.makedirs(outputBaseFile, exist_ok=True)
    print("Corresponding segmentation ground truth will be saved in {}".format(outFile))

    images = []
    annotations = []
    instance_count = 0
    for progress, f in enumerate(cityscapes_files):
        originalFormat = np.array(Image.open(f))

        fileName = os.path.basename(f)
        cityscapes_imageId = fileName.replace("_gtFine_instanceIds.png", "")
        cityscapes_inputFileName = fileName.replace("_gtFine_instanceIds.png", "_leftImg8bit.png")
        city_name = cityscapes_inputFileName.split("_")[0]
        cityscapes_img_path = os.path.join(cityscapesPath, "leftImg8bit", cityscapesSplit, city_name, cityscapes_inputFileName)
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


        city_height = 1024
        city_width = 2048
        coco_masked_image = np.zeros((city_height, city_width, 3), dtype="uint8")
        cityscapes_masked_image = np.ones((city_height, city_width, 3), dtype="uint8")

        coco_masked_panoptic = np.zeros((city_height, city_width, 3), dtype="uint8")
        cityscapes_masked_panoptic = np.ones((city_height, city_width, 3), dtype="uint8")

        roadPixelID = 0 if useTrainId else 7
        road_pixels = np.where(originalFormat == roadPixelID)

        image_instance = coco_instances[progress]

        for k, grouped_instance in enumerate(image_instance):
            bbox = grouped_instance["bbox"]

            Image.fromarray(grouped_instance["panoptic_mask"][bbox[0]: bbox[2], bbox[1]: bbox[3]]).save(
               os.path.join(outputFolder, "panoptic.png"))
            Image.fromarray(grouped_instance["image_mask"][bbox[0]: bbox[2], bbox[1]: bbox[3]]).save(os.path.join(outputFolder, "image.png"))
            Image.fromarray(grouped_instance["semantic_mask"][bbox[0]: bbox[2], bbox[1]: bbox[3]]).save(os.path.join(outputFolder, "semantic.png"))

            coco_panoptic = grouped_instance["panoptic_mask"][bbox[0]: bbox[2], bbox[1]: bbox[3]]
            coco_image = grouped_instance["image_mask"][bbox[0]: bbox[2], bbox[1]: bbox[3]]
            coco_semantic = grouped_instance["semantic_mask"][bbox[0]: bbox[2], bbox[1]: bbox[3]]
            coco_h, coco_w = coco_panoptic.shape
            coco_panoptic_format = np.zeros((coco_h, coco_w, 3), dtype="uint8")
            coco_image_format = np.zeros((coco_h, coco_w, 3), dtype="uint8")

            end_width = city_width - coco_w

            road_end_h = np.where(road_pixels[0] > coco_h)
            road_end_w = np.where(road_pixels[1] < end_width)

            road_eligible_pixels = np.intersect1d(road_end_h, road_end_w)

            road_random_pixel = random.randint(0, len(road_eligible_pixels))

            start_width = road_pixels[1][road_eligible_pixels[road_random_pixel]]
            start_height = road_pixels[0][road_eligible_pixels[road_random_pixel]] - coco_h


            for coco_instance_id in grouped_instance["instance_ids"]:

                segmentId = coco_instance_id

                '''end_width = city_width - w
                end_height = city_height - h
                start_width = random.randint(0, end_wisdth)
                start_height = random.randint(0, end_height)'''

                categoryId = np.max(coco_semantic)
                isCrowd = 0

                mask = coco_panoptic == segmentId
                color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                coco_panoptic_format[mask] = color

                coco_image_format[mask] = coco_image[mask]

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
                # add start width and height to accommodate the bounding box positions in original cityscapes size
                bbox = [start_width + int(x), start_height+ int(y), int(width), int(height)]

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


            # extract only OOD object pixels
            # clear out if already existing ood instances in this location
            coco_masked_image[start_height:start_height + coco_h, start_width:start_width + coco_w, :] = np.zeros(coco_image_format.shape)
            coco_masked_image[start_height:start_height + coco_h, start_width:start_width + coco_w, :] += coco_image_format


            # same steps for panoptic mask creation
            coco_masked_panoptic[start_height:start_height + coco_h, start_width:start_width + coco_w, :] = np.zeros(coco_panoptic_format.shape)
            coco_masked_panoptic[start_height:start_height + coco_h, start_width:start_width + coco_w, :] += coco_panoptic_format


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