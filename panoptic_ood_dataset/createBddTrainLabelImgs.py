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
def convert2TrainIds(bddPath=None, setNames=["val", "train", "test"]):
    # Where to look for Cityscapes
    if bddPath is None:
        raise "bddPath cannot be None"


    for setName in setNames:
        # how to search for all ground truth
        searchFine   = os.path.join(bddPath, "gtFine", setName, "*" , "*_gtFine_labelIds.png")
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
        print("Converting {} labelIDs to TrainId files for {} set.".format(len(files), setName))
        images = []
        annotations = []
        for progress, f in enumerate(files):
            fileName = os.path.basename(f)
            city_name = fileName.split("_")[0]
            outputFileName = fileName.replace("_labelIds.png", "_labelTrainIds.png")
            originalFormat = np.array(Image.open(f))

            train_format = np.ones(
                (originalFormat.shape[0], originalFormat.shape[1], 3), dtype=np.uint8
            ) * 255

            semanticIds = np.unique(originalFormat)
            for semanticId in semanticIds:
                if semanticId == 50:
                    train_format[originalFormat==semanticId] = 19
                else:
                    train_format[originalFormat==semanticId] = labels[semanticId].trainId

            Image.fromarray(train_format).save(os.path.join(bddPath, "gtFine", setName, city_name, outputFileName))

            print("\rProgress: {:>3.2f} %".format((progress + 1) * 100 / len(files)), end=' ')
            sys.stdout.flush()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd-path",
                        dest="bddPath",
                        help="path to the BDD dataset images which are converted to cityscapes format",
                        default="/home/mohan/kiran/bdd_ood_val/filtered_bdd_filtered/cityscapes/",
                        type=str)
    parser.add_argument("--set-names",
                        dest="setNames",
                        help="set names to which apply the function to",
                        nargs='+',
                        default=["val"],
                        type=str)
    args = parser.parse_args()

    convert2TrainIds(args.bddPath, args.setNames)


# call the main
if __name__ == "__main__":
    main()
