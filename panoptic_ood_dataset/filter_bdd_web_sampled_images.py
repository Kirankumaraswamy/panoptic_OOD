import random

import cv2
import argparse
import os
import glob
import shutil
import json
import numpy as np

sampled_image_count = 6

def display_to_filter_images(args):
    bddOODFolder = args.bddOODFolder
    filteredPath = args.filteredPath
    bddSplit = args.bddSplit
    mode = "gtFine"

    if not os.path.exists(bddOODFolder):
        raise "Bdd OOD folder doesn't exist"

    if not os.path.exists(filteredPath):
        raise "Filter path doesn't exist"

    bdd_ood_json_file = os.path.join(bddOODFolder, mode, "bdd_panoptic_"+bddSplit+".json")
    panoptic_gt = json.load(open(bdd_ood_json_file))

    # construct image_id to index location in grountruth json
    id_to_index_map = {}
    for i, image in enumerate(panoptic_gt["images"]):
        id = image["bdd_id"]
        id_to_index_map[id] = i

    base_name = os.path.basename(filteredPath)
    saved_instance_dir = os.path.join(filteredPath, "filtered_"+base_name, "bdd_ood")
    saved_instance_json_file = os.path.join(saved_instance_dir, mode, "bdd_panoptic_"+bddSplit+".json")

    if not os.path.exists(saved_instance_dir):
        os.makedirs(saved_instance_dir)
        os.makedirs(os.path.join(saved_instance_dir, "leftImg8bit"))

    if not os.path.exists(os.path.join(saved_instance_dir, mode, "bdd_panoptic_" + bddSplit)):
        os.makedirs(os.path.join(saved_instance_dir, mode, "bdd_panoptic_" + bddSplit))
        os.makedirs(os.path.join(saved_instance_dir, mode, bddSplit), exist_ok=True)

    if not os.path.exists(os.path.join(saved_instance_dir, "leftImg8bit", bddSplit)):
        os.makedirs(os.path.join(saved_instance_dir, "leftImg8bit", bddSplit))

    if not os.path.exists(saved_instance_json_file):
        data = {}
        data["annotations"] = []
        data["images"] = []
        data["categories"] = panoptic_gt["categories"]
        with open(saved_instance_json_file, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)


    saved_instance_list_file = os.path.join(saved_instance_dir, "saved_files_"+bddSplit+".txt")

    if not os.path.exists(saved_instance_list_file):
        open(saved_instance_list_file, 'w').close()

    # get saved json content
    saved_instance_gt = json.load(open(saved_instance_json_file))

    # get imaged ids of saved instances
    with open(saved_instance_list_file, 'r') as f:
        lines = f.readlines()
    saved_instances_list = [line.split("\t")[0].strip() for line in lines]

    # get image IDs of all bdd images
    bdd_images = panoptic_gt["images"]
    bdd_image_ids = [image["bdd_id"] for image in bdd_images]

    # get IDs of instances to be filtered out
    non_filtered_list = list(set(bdd_image_ids) - set(saved_instances_list))

    key_to_image_map = {"49": 0, "50": 1, "51": 2, "52":3, "53": 4, "54": 5, "55": 6, "56":7, "57":8}

    print("Images to filter: ", len(non_filtered_list))
    while len(non_filtered_list) > 0:
        image_id = non_filtered_list[0]
        cityname = image_id.split("_")[0]

        rgb_image_folder = os.path.join(saved_instance_dir, "leftImg8bit", bddSplit, cityname)
        if not os.path.exists(rgb_image_folder):
            os.makedirs(rgb_image_folder)
            os.makedirs(os.path.join(saved_instance_dir, mode, bddSplit, cityname))

        '''sampled_images = []
        for i in range(sampled_image_count):
            img = cv2.imread(os.path.join(bddOODFolder, "leftImg8bit", bddSplit, cityname,
                                           image_id + "_leftImg8bit_"+str(i)+".png"))
            img = cv2.resize(img, dsize=(512, 256), interpolation=cv2.INTER_CUBIC)
            sampled_images.append(img)

        horizontal_1 = np.concatenate((sampled_images[0], sampled_images[1], sampled_images[2]), axis=1)
        horizontal_2 = np.concatenate((sampled_images[3], sampled_images[4], sampled_images[5]), axis=1)
        #horizontal_3 = np.concatenate((sampled_images[6], sampled_images[7], sampled_images[8]), axis=1)
        verticle = np.concatenate((horizontal_1, horizontal_2), axis=0)

        cv2.imshow("Image order 1, 2, 3 in the top  and 4, 5, 6 at middle and 7, 8, 9 at bottom", verticle)'''

        while True:
            k = int(list(key_to_image_map.keys())[0])
            #k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                cv2.destroyAllWindows()
                break
            elif str(k) in key_to_image_map.keys():
                saved_id = non_filtered_list.pop(0)

                image_no = key_to_image_map[str(k)]
                print("Image {} saved with sample ".format((saved_id, image_no+1)))

                from_copy_panoptic_file = os.path.join(bddOODFolder, mode, "bdd_panoptic_" + bddSplit, image_id + "_gtFine_panoptic_"+str(image_no)+".png")
                from_copy_rgb_file = os.path.join(bddOODFolder, "leftImg8bit", bddSplit, cityname,
                                       image_id + "_leftImg8bit_"+str(image_no)+".png")
                from_copy_labelIds = os.path.join(bddOODFolder, mode, bddSplit, cityname,
                                       image_id + "_gtFine_labelIds_"+str(image_no)+".png")
                from_copy_instanceIds = os.path.join(bddOODFolder, mode, bddSplit, cityname,
                                       image_id + "_gtFine_instanceIds_"+str(image_no)+".png")

                to_copy_panoptic_file = os.path.join(saved_instance_dir, mode,
                                                       "bdd_panoptic_" + bddSplit,
                                                       image_id + "_gtFine_panoptic.png")
                to_copy_rgb_file = os.path.join(saved_instance_dir, "leftImg8bit", bddSplit, cityname,
                                                  image_id + "_leftImg8bit.png")
                to_copy_labelIds = os.path.join(saved_instance_dir, mode, bddSplit, cityname,
                                                  image_id + "_gtFine_labelIds.png")
                to_copy_instanceIds = os.path.join(saved_instance_dir, mode, bddSplit, cityname,
                                                     image_id + "_gtFine_instanceIds.png")



                shutil.copyfile(from_copy_panoptic_file, to_copy_panoptic_file)
                shutil.copyfile(from_copy_rgb_file, to_copy_rgb_file)
                shutil.copyfile(from_copy_labelIds, to_copy_labelIds)
                shutil.copyfile(from_copy_instanceIds, to_copy_instanceIds)

                saved_instance_gt["images"].append(panoptic_gt["images"][id_to_index_map[saved_id]])
                saved_instance_gt["annotations"].append(panoptic_gt["annotations"][id_to_index_map[saved_id]][image_no])

                with open(saved_instance_list_file, 'a') as f:
                    line = saved_id+ "\t" + str(image_no+1)
                    f.write(line)
                    f.write("\n")
                '''with open(saved_instance_json_file, 'w') as f:
                    json.dump(saved_instance_gt, f, sort_keys=True, indent=4)'''

                cv2.destroyAllWindows()
                break
    with open(saved_instance_json_file, 'w') as f:
        json.dump(saved_instance_gt, f, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdd-ood-folder",
                        dest="bddOODFolder",
                        help="path to the Bdd OOD sampled instances",
                        default="/home/kumarasw/OOD_dataset/web_ood_rohit/bdd_ood",
                        type=str)
    parser.add_argument("--filtered-path",
                        dest="filteredPath",
                        help="path to save filters instances",
                        default="/home/kumarasw/OOD_dataset/web_ood_rohit/bdd_train_filtered",
                        type=str)
    parser.add_argument("--bdd-split",
                        dest="bddSplit",
                        help="bdd data split to be used to create the OOD dataset",
                        default="train",
                        type=str)

    args = parser.parse_args()

    display_to_filter_images(args)


# call the main
if __name__ == "__main__":
    main()

