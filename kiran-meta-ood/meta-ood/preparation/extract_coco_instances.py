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

coco_categories = {
    "airplane": ("sky", "car", 1),
    "boat": ("road", "car", 1),
    "bird": ("road", "person", 0.3),
    "cat": ("road", "person", 0.3),
    "dog": ("road", "person", 1),
    "horse": ("road", "car", 1.5),
    "sheep": ("road", "person", 1.5),
    "cow": ("road", "person", 2),
    "elephant": ("road", "car", 3),
    "bear": ("road", "person", 1.5),
    "zebra": ("road", "car", 1),
    "giraffe": ("road", "car", 2),
    "skateboard": ("road", "person", 0.3),
    "suitcase": ("road", "car", 0.2),
    "bottle": ("road", "person", 0.1),
    "chair": ("road", "car", 0.3),
    "couch": ("road", "car", 1),
    "teddy bear": ("road", "person", 0.5)
}

ood_train_id = 20
ood_id = 50

# The main method
def extract_instances(cityscapesPath=None, cityscapesSplit=None, cocoPath=None, cocoSplit=None, cocoYear=None, outputFolder=None, useTrainId=False):
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

    coco_instances_save_dir = os.path.join(outputFolder, "coco_instances")
    coco_images_save_folder = "coco_rgb"
    coco_panoptic_save_folder = "coco_panoptic"
    coco_semantic_save_folder = "coco_semantic"

    print("Creating folder {} for saving OOD images and ground truth".format(coco_instances_save_dir))
    os.makedirs(os.path.join(coco_instances_save_dir, coco_images_save_folder), exist_ok=True)
    os.makedirs(os.path.join(coco_instances_save_dir, coco_panoptic_save_folder), exist_ok=True)
    os.makedirs(os.path.join(coco_instances_save_dir, coco_semantic_save_folder), exist_ok=True)


    coco_images_dir = '{}/{}'.format(cocoPath, cocoSplit + str(cocoYear))
    annotation_file = '{}/annotations/instances_{}.json'.format(cocoPath, cocoSplit + str(cocoYear))
    coco_gt_json_file = os.path.join(cocoPath, "annotations", "panoptic_" + cocoSplit + cocoYear + ".json")
    json_data = json.load(open(coco_gt_json_file))
    coco_instances = []
    output_json = "coco_instances.json"

    tools = coco_tools(annotation_file)
    include_cat_Ids = tools.getCatIds(catNms=coco_categories)
    coco_name_id_map = {}
    for cat in tools.dataset["categories"]:
        coco_name_id_map[cat["id"]] = cat["name"]

    coco_include_img_Ids = []
    for cat_Id in include_cat_Ids:
        coco_include_img_Ids += tools.getImgIds(catIds=cat_Id)
    coco_include_img_Ids = set(coco_include_img_Ids)
    coco_img_Ids = [int(image[:-4]) for image in os.listdir(coco_images_dir) if int(image[:-4]) in coco_include_img_Ids]
    coco_img_Ids.sort()

    print("Total coco images selected: ", len(coco_img_Ids))

    selected_coco_img_Ids = []

    total_instances = 0
    instance_count = 1
    coco_instance_annotations = {}
    print("Total images: ", len(coco_img_Ids))
    for i, img_Id in enumerate(coco_img_Ids):
        print("Image: ", i)
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
            coco_semantic_id = annotations[j]["category_id"]
            box = annotations[j]["bbox"]
            # h_min, w_min. h_max, w_max
            bbox = [int(box[1]), int(box[0]), int(box[1])+int(box[3]), int(box[0])+int(box[2])]
            if coco_semantic_id in include_cat_Ids:
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
                            "semantic_ids": [coco_semantic_id],
                            "semantic_names": [coco_name_id_map[coco_semantic_id]]
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
                            semantic_names = value["semantic_names"]
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
                                semantic_names.append(coco_name_id_map[coco_semantic_id])

                                # object bounding box
                                min_h = np.min(combined_mask_pos[0])
                                min_w = np.min(combined_mask_pos[1])
                                max_h = np.max(combined_mask_pos[0])
                                max_w = np.max(combined_mask_pos[1])
                                bbox = [int(min_h), int(min_w), int(max_h), int(max_w)]

                                grouped_instances[key] = {
                                    "bbox": bbox,
                                    "panoptic_mask": p_mask,
                                    "semantic_mask": s_mask,
                                    "image_mask": coco_image_mask,
                                    "instance_ids": instance_ids,
                                    "semantic_ids": semantic_ids,
                                    "semantic_names": semantic_names,
                                }
                                instance_count += 1

                        if not overlap:
                            grouped_instances[count] = {
                                "bbox": bbox,
                                "panoptic_mask": coco_panoptic_mask,
                                "semantic_mask": coco_semantic_mask,
                                "image_mask": coco_image_mask,
                                "instance_ids": [instance_id],
                                "semantic_ids": [categoryId],
                                "semantic_names": [coco_name_id_map[coco_semantic_id]]
                            }
                            count += 1
                            instance_count += 1

        for key, value in grouped_instances.items():
            image_path = os.path.join(coco_instances_save_dir, coco_images_save_folder, str(img_Id) +"_" + str(key+1) + "_rgb.png")
            panoptic_path = os.path.join(coco_instances_save_dir, coco_panoptic_save_folder, str(img_Id) + "_" + str(key + 1) + "_panoptic.png")
            sematic_path = os.path.join(coco_instances_save_dir, coco_semantic_save_folder, str(img_Id) + "_" + str(key + 1) + "_semantic.png")
            Image.fromarray(value["image_mask"]).save(image_path)
            Image.fromarray(value["panoptic_mask"]).save(panoptic_path)
            Image.fromarray(value["semantic_mask"]).save(sematic_path)


            coco_instance_annotations[str(img_Id) +"_" + str(key+1)] = { "bbox": value["bbox"],
                                                                    "instance_ids": value["instance_ids"],
                                                                    "semantic_names": value["semantic_names"],
                                                                         "rgb_image": image_path,
                                                                         "panoptic_image": panoptic_path,
                                                                         "semantic_image": sematic_path}
            total_instances += 1


        '''if total_instances >= 200:
            break'''

    print("Total Instances: ", total_instances)

    outFile = os.path.join(outputFolder, output_json)
    print("\nSaving the json file {}".format(outFile))

    with open(outFile, 'w') as f:
        json.dump(coco_instance_annotations, f, sort_keys=True, indent=4)





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

    args = parser.parse_args()

    extract_instances(args.cityscapesPath, args.cityscapesSplit, args.cocoPath, args.cocoSplit, args.cocoYear, args.outputFolder, args.useTrainId)


# call the main
if __name__ == "__main__":
    main()