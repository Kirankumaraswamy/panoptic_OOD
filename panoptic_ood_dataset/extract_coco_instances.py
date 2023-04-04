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
    "airplane": ("sky", "car", 0.7, "train"),
    "bird": ("sky", "person", 0.5, "train"),
    "dog": ("road", "person", 0.6, "train"),
    "horse": ("road", "car", 1, "train"),
    "elephant": ("road", "car", 1.5, "train"),
    "zebra": ("road", "car", 0.6, "train"),
    "giraffe": ("road", "car", 1.3, "train"),
    "suitcase": ("road", "car", 0.2, "train"),
    "bottle": ("road", "person", 0.3, "train"),
    "chair": ("road", "car", 0.3, "train"),
    "teddy bear": ("road", "person", 0.4, "train"),
    "vase": ("road", "person", 0.4, "train"),
    "microwave": ("road", "car", 0.3, "train"),
    "umbrella": ("road", "car", 0.4, "train"),
    "skateboard": ("road", "person", 0.4, "train"),
    "toilet": ("road", "person", 0.7, "train"),

    "boat": ("road", "car", 1, "test"),
    "cat": ("road", "person", 0.3, "test"),
    "sheep": ("road", "person", 1.5, "test"),
    "cow": ("road", "person", 2, "test"),
    "bear": ("road", "person", 1.5, "test"),
    "couch": ("road", "car", 1, "test"),
    "kite": ("sky", "person", 0.4, "test"),
    "tv": ("road", "car", 0.5, "test"),
    "cup": ("road", "person", 0.4, "test"),
    "toaster": ("road", "person", 0.4, "test"),
    "surfboard": ("road", "person", 0.4, "test")
}

'''coco_categories = {
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
    "bird": ("road", "car", 0.05, "test"),
    "bear": ("road", "car", 0.5, "test"),
    "cow": ("road", "car", 1, "test"),
    "tiger": ("road", "car", 0.5, "test"),
    "fire hydrant": ("road", "car", 0.2, "test"),
    "portable public toilet": ("road", "car", 1, "test"),
    "drone": ("road", "car", 0.1, "test"),
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
    "tortoise": ("road", "car", 0.1, "test"),
    "unknown": ("road", "car", 0.4, "train")
}
'''
ood_train_id = 19
ood_id = 50

# The main method
def extract_instances(args):
    cocoPath = args.cocoPath
    cocoSplit = args.cocoSplit
    cocoSplit = args.cocoSplit
    cocoYear = args.cocoYear
    outputFolder = args.outputFolder
    oodSplit = args.oodSplit
    useTrainId = args.useTrainId

    if outputFolder is None:
        raise "Please specify output directory."

    if oodSplit != "train" and oodSplit != "test":
        raise "COCO split value can be either train or test."


    coco_instances_save_dir = os.path.join(outputFolder, oodSplit)


    coco_images_save_folder = "coco_rgb"
    coco_panoptic_save_folder = "coco_panoptic"
    coco_semantic_save_folder = "coco_semantic"

    print("Creating folder {} for saving OOD images and ground truth".format(coco_instances_save_dir))

    # for training instances
    os.makedirs(os.path.join(coco_instances_save_dir, coco_images_save_folder), exist_ok=True)
    os.makedirs(os.path.join(coco_instances_save_dir, coco_panoptic_save_folder), exist_ok=True)
    os.makedirs(os.path.join(coco_instances_save_dir, coco_semantic_save_folder), exist_ok=True)



    coco_images_dir = '{}/{}'.format(cocoPath, cocoSplit + str(cocoYear))
    annotation_file = '{}/annotations/instances_{}.json'.format(cocoPath, cocoSplit + str(cocoYear))
    #coco_gt_json_file = os.path.join(cocoPath, "annotations", "panoptic_" + cocoSplit + cocoYear + ".json")
    #json_data = json.load(open(coco_gt_json_file))
    coco_instances = []
    output_json = "coco_instances.json"

    tools = coco_tools(annotation_file)

    include_coco_categories = []
    for key in coco_categories.keys():
        if coco_categories[key][3] == oodSplit:
            include_coco_categories.append(key)
    include_cat_Ids = tools.getCatIds(catNms=include_coco_categories)
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

        image_instance_count = 1
        for j in range(len(annotations)):
            coco_semantic_id = annotations[j]["category_id"]
            box = annotations[j]["bbox"]
            # h_min, w_min. h_max, w_max
            bbox = [int(box[1]), int(box[0]), int(box[1])+int(box[3]), int(box[0])+int(box[2])]
            if coco_semantic_id in include_cat_Ids:
                categoryId = ood_train_id if useTrainId else ood_id
                instance_id = categoryId * 1000 + image_instance_count


                mask = tools.annToMask(annotations[j])
                mask_position = np.where(mask > 0)
                area = int(np.sum(mask > 0))

                if area < 3000:
                    continue

                if ((np.max(mask_position[0]) > h - 5) or (np.max(mask_position[1]) > w - 5)
                        or (np.min(mask_position[0]) < 5) or (np.min(mask_position[1]) < 5)):
                    continue

                coco_panoptic_mask = mask * instance_id
                coco_semantic_mask = mask * categoryId

                coco_image_mask = np.expand_dims(mask, axis=2) * coco_rgb_img
                image_instance_count += 1

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

            # some images has objects inside objects. Discard such images. Eg: teddy bear inside  chair
            instance_ids = value["instance_ids"]
            panoptic_values = np.unique(value["panoptic_mask"])
            id_found = True
            for id in instance_ids:
                if id not in panoptic_values:
                    id_found = False
            if not id_found:
                break

            rgb_sub_path = os.path.join(coco_images_save_folder, str(img_Id) +"_" + str(key+1) + "_rgb.png")
            panoptic_sub_path = os.path.join(coco_panoptic_save_folder, str(img_Id) + "_" + str(key + 1) + "_panoptic.png")
            semantic_sub_path = os.path.join(coco_semantic_save_folder, str(img_Id) + "_" + str(key + 1) + "_semantic.png")
            image_path = os.path.join(coco_instances_save_dir, rgb_sub_path)
            panoptic_path = os.path.join(coco_instances_save_dir, panoptic_sub_path)
            sematic_path = os.path.join(coco_instances_save_dir, semantic_sub_path)
            Image.fromarray(value["image_mask"]).save(image_path)
            Image.fromarray(value["panoptic_mask"]).save(panoptic_path)
            Image.fromarray(value["semantic_mask"]).save(sematic_path)


            coco_instance_annotations[str(img_Id) +"_" + str(key+1)] = { "bbox": value["bbox"],
                                                                    "instance_ids": value["instance_ids"],
                                                                    "semantic_names": value["semantic_names"],
                                                                         "rgb_image": os.path.join(oodSplit, rgb_sub_path),
                                                                         "panoptic_image": os.path.join(oodSplit, panoptic_sub_path),
                                                                         "semantic_image": os.path.join(oodSplit, semantic_sub_path)}
            total_instances += 1

        print("\rProgress: {:>3.2f} %".format((i + 1) * 100 / len(coco_img_Ids)), end=' ')
        '''if total_instances >= 100:
            break'''

    print("Total Instances: ", total_instances)

    outFile = os.path.join(coco_instances_save_dir, output_json)
    print("\nSaving the json file {}".format(outFile))

    with open(outFile, 'w') as f:
        json.dump(coco_instance_annotations, f, sort_keys=True, indent=4)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-folder",
                        dest="cocoPath",
                        help="path to the COCO dataset folder",
                        default="/export/kiran/coco/2017",
                        type=str)
    parser.add_argument("--coco-split",
                        dest="cocoSplit",
                        help="coco data split to be used to create the OOD dataset",
                        default="train",
                        type=str)
    parser.add_argument("--coco-year",
                        dest="cocoYear",
                        help="coco dataset year to be used to create the OOD dataset",
                        default="2017",
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the output folder.",
                        default="/home/kumarasw/OOD_dataset/coco_instances",
                        type=str)

    parser.add_argument("--ood-split",
                        dest="oodSplit",
                        help="Split for OOD objects. train or test",
                        default="train",
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")

    args = parser.parse_args()

    extract_instances(args)


# call the main
if __name__ == "__main__":
    main()