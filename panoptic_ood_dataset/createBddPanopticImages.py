#!/usr/bin/python
#
# Converts the *instanceIds.png annotations of the Cityscapes dataset
# to COCO-style panoptic segmentation format (http://cocodataset.org/#format-data).
# The convertion is working for 'fine' set of the annotations.
#
# By default with this tool uses IDs specified in labels.py. You can use flag
# --use-train-id to get train ids for categories. 'ignoreInEval' categories are
# removed during the conversion.
#
# In panoptic segmentation format image_id is used to match predictions and ground truth.
# For cityscapes image_id has form <city>_123456_123456 and corresponds to the prefix
# of cityscapes image files.
#

# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image
import cv2
# cityscapes imports
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.helpers.labels import id2label, labels

# BDD panoptic mapping
'''0:  unlabeled
1:  dynamic
2:  ego vehicle
3:  ground
4:  static
5:  parking
6:  rail track
7:  road
8:  sidewalk
9:  bridge
10: building
11: fence
12: garage
13: guard rail
14: tunnel
15: wall
16: banner
17: billboard
18: lane divider
19: parking sign
20: pole
21: polegroup
22: street light
23: traffic cone
24: traffic device
25: traffic light
26: traffic sign
27: traffic sign frame
28: terrain
29: vegetation
30: sky
31: person
32: rider
33: bicycle
34: bus
35: car
36: caravan
37: motorcycle
38: trailer
39: train
40: truck'''


bdd_cityscapes_label_map = {
0:  0,
1:  5,
2:  1,
3:  6,
4:  4,
5:  9,
6:  10,
7:  7,
8:  8,
9:  15,
10: 11,
11: 13,
12: 0,
13: 14,
14: 16,
15: 12,
16: 0,
17: 0,
18: 0,
19: 0,
20: 17,
21: 18,
22: 0,
23: 0,
24: 0,
25: 19,
26: 20,
27: 0,
28: 22,
29: 21,
30: 23,
31: 24,
32: 25,
33: 33,
34: 28,
35: 26,
36: 29,
37: 32,
38: 30,
39: 31,
40: 27
}


# The main method
def convert2panoptic(bddPath=None, jsonFile=None, outputFolder=None, useTrainId=False, setNames=["val", "train", "test"]):
    # Where to look for Cityscapes
    if bddPath is None:
        raise "bddPath cannot be None"
    if jsonFile is None:
        raise "jsonFile cannot be None"

    if outputFolder is None:
        raise "outputFolder cannot be None"

    bdd_json_data = json.load(open(jsonFile))
    annotations = bdd_json_data['annotations']
    annotation_map = {}
    for annotation in annotations:
        id = annotation["file_name"].split(os.sep)[-1].split(".png")[0]
        annotation_map[id] = annotation

    categories = []
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                           'name': label.name,
                           'color': label.color,
                           'supercategory': label.category,
                           'isthing': 1 if label.hasInstances else 0})

    for setName in setNames:
        os.makedirs(os.path.join(outputFolder, "bdd", "gtFine", setName, "city"), exist_ok=True)
        os.makedirs(os.path.join(outputFolder, "bdd", "leftImg8bit", setName, "city"), exist_ok=True)
        # how to search for all ground truth
        searchFine   = os.path.join(bddPath, "labels", "pan_seg", "colormaps", setName, "*.png")
        # search files
        filesFine = glob.glob(searchFine)
        filesFine.sort()

        files = filesFine
        # quit if we did not find anything
        if not files:
            printError(
                "Did not find any files for {} set using matching pattern {}. Please consult the README.".format(setName, searchFine)
            )
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

        trainIfSuffix = "_trainId" if useTrainId else ""
        outputBaseFile = "bdd_panoptic_{}{}".format(setName, trainIfSuffix)
        outFile = os.path.join(outputFolder, "bdd", "gtFine", "{}.json".format(outputBaseFile))
        print("Json file with the annotations in panoptic format will be saved in {}".format(outFile))
        panopticFolder = os.path.join(outputFolder, "bdd", "gtFine", outputBaseFile)
        if not os.path.isdir(panopticFolder):
            print("Creating folder {} for panoptic segmentation PNGs".format(panopticFolder))
            os.mkdir(panopticFolder)
        print("Corresponding segmentations in .png format will be saved in {}".format(panopticFolder))

        images = []
        annotations = []
        for progress, f in enumerate(files):
            city_name = "city"
            fileName = os.path.basename(f)
            imageId = fileName.replace(".png", "")
            originalFormat = np.array(Image.open(f))
            rgb_image_file = os.path.join(bddPath, "images", "10k", setName, imageId+".jpg")
            rgb_image = np.array(Image.open(rgb_image_file))

            originalFormat = cv2.resize(originalFormat, dsize=(2048, 1024),
                                       interpolation=cv2.INTER_NEAREST)
            rgb_image = cv2.resize(rgb_image, dsize=(2048, 1024),
                                       interpolation=cv2.INTER_NEAREST)

            inputFileName = city_name+"_"+imageId+ "_0_leftImg8bit.png"
            outputFileName = city_name + "_"+ imageId+ "_0_gtFine"+ "_panoptic.png"
            labelIdsName = city_name + "_"+ imageId + "_0_gtFine"+ "_labelIds.png"
            instanceIdsName = city_name + "_"+ imageId + "_0_gtFine"+ "_instanceIds.png"

            # image entry, id for image is its filename without extension
            images.append({"id": imageId,
                           "width": int(originalFormat.shape[1]),
                           "height": int(originalFormat.shape[0]),
                           "file_name": inputFileName})

            pan_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
            )

            semantic_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1]), dtype=np.uint8
            )

            instance_format = np.zeros(
                (originalFormat.shape[0], originalFormat.shape[1]), dtype=np.uint32
            )

            segmentIds = np.unique(originalFormat)
            bdd_seg_info = annotation_map[imageId]['segments_info']
            bdd_seg_info_map = {}
            for seg in bdd_seg_info:
                seg_id = seg["id"]
                bdd_seg_info_map[seg_id] = seg
            instance_count_map = {}

            segmInfo = []
            for segmentId in segmentIds:
                # donot consider unlabelled pixels
                if segmentId == 0:
                    continue

                if bdd_seg_info_map[segmentId]["iscrowd"]:
                    isCrowd = 1
                else:
                    isCrowd = 0
                semanticId = bdd_seg_info_map[segmentId]["category_id"]

                # get cityscapes ID
                semanticId = bdd_cityscapes_label_map[semanticId]
                labelInfo = id2label[semanticId]
                categoryId = labelInfo.trainId if useTrainId else labelInfo.id
                if labelInfo.ignoreInEval:
                    continue
                if not labelInfo.hasInstances:
                    isCrowd = 0

                if labelInfo.hasInstances:
                    instance_count = 0
                    if categoryId in instance_count_map:
                        instance_count = instance_count_map[categoryId]
                        instance_count += 1
                    instance_count_map[categoryId] = instance_count

                    encoded_segmentId = categoryId * 1000 + instance_count
                else:
                    encoded_segmentId = categoryId

                mask = originalFormat == segmentId
                color = [encoded_segmentId % 256, encoded_segmentId // 256, encoded_segmentId // 256 // 256]
                pan_format[mask] = color
                semantic_format[mask] = categoryId
                instance_format[mask] = encoded_segmentId

                area = np.sum(mask) # segment area computation

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

                segmInfo.append({"id": int(encoded_segmentId),
                                 "category_id": int(categoryId),
                                 "area": int(area),
                                 "bbox": bbox,
                                 "iscrowd": isCrowd})

            annotations.append({'image_id': city_name+"_"+imageId+"_0",
                                'file_name': outputFileName,
                                "segments_info": segmInfo})

            Image.fromarray(pan_format).save(os.path.join(panopticFolder, outputFileName))
            Image.fromarray(semantic_format).save(os.path.join(outputFolder, "bdd", "gtFine", setName,city_name, labelIdsName))
            Image.fromarray(instance_format).save(os.path.join(outputFolder, "bdd", "gtFine", setName,city_name, instanceIdsName))

            Image.fromarray(rgb_image).save(os.path.join(outputFolder, "bdd", "leftImg8bit", setName,city_name, inputFileName))


            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()


        print("\nSaving the json file {}".format(outFile))
        d = {'images': images,
             'annotations': annotations,
             'categories': categories}
        with open(outFile, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-folder",
                        dest="bddColorMapsPath",
                        help="path to the BDD dataset images and labels folder",
                        default="/home/kumarasw/OOD_dataset/BDD/bdd100k",
                        type=str)
    parser.add_argument("--json-file",
                        dest="jsonFile",
                        help="path to the BDD dataset ground truth json file converted to coco format",
                        default="/home/kumarasw/OOD_dataset/BDD/bdd100k/labels/pan_seg/bdd_panoptic.json",
                        type=str)
    parser.add_argument("--output-folder",
                        dest="outputFolder",
                        help="path to the folder ",
                        default="/home/kumarasw/OOD_dataset/bdd",
                        type=str)
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["val", "train"],
                        type=str)
    args = parser.parse_args()

    convert2panoptic(args.bddColorMapsPath, args.jsonFile, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
