import cv2
import argparse
import os
import glob
import shutil
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

num_map = {
    "1": "./one.png",
    "2": "./two.png",
    "3": "./three.png",
    "4": "./four.png",
    "5": "./five.png",
    "6": "./six.png",
    "10": "./ten.png"
}

def display_to_filter_images(args):
    oodExtractFolder = args.oodExtractFolder
    filteredPath = args.filteredPath
    oodSplit = args.oodSplit

    if not os.path.exists(oodExtractFolder):
        raise "ood extract folder doesn't exist"

    if not os.path.exists(filteredPath):
        raise "Filter path doesn't exist"

    if oodSplit != "train" and oodSplit != "test":
        raise "COCO split value can be either train or test."

    ood_instance_files_dir = os.path.join(oodExtractFolder, oodSplit)

    filter_parent_dir = "filtered_ood_instances"

    saved_instance_dir = os.path.join(filteredPath, filter_parent_dir, oodSplit)
    saved_instance_json_file = os.path.join(filteredPath, filter_parent_dir, oodSplit, "ood_instances.json")

    if not os.path.exists(saved_instance_dir):
        os.makedirs(saved_instance_dir)
        os.makedirs(os.path.join(saved_instance_dir, "ood_panoptic"))
        os.makedirs(os.path.join(saved_instance_dir, "ood_rgb"))
        os.makedirs(os.path.join(saved_instance_dir, "ood_semantic"))

    if not os.path.exists(saved_instance_json_file):
        data = {}
        with open(saved_instance_json_file, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4)

    saved_instance_list_file = os.path.join(filteredPath, filter_parent_dir, oodSplit, "saved_files.txt")
    deleted_instance_list_file = os.path.join(filteredPath, filter_parent_dir, oodSplit, "deleted_files.txt")

    if not os.path.exists(saved_instance_list_file):
        open(saved_instance_list_file, 'w').close()

    if not os.path.exists(deleted_instance_list_file):
        open(deleted_instance_list_file, 'w').close()

    # get saved json content
    saved_instance_gt = json.load(open(saved_instance_json_file))

    # get imaged ids of saved instances
    with open(saved_instance_list_file, 'r') as f:
        lines = f.readlines()
    saved_instances_list = [line.strip() for line in lines]

    # get imaged ids of deleted instances
    with open(deleted_instance_list_file, 'r') as f:
        lines = f.readlines()
    deleted_instances_list = [line.strip() for line in lines]

    # get image IDs of all extracted instances
    ood_extract_rgb_dir = oodExtractFolder
    ood_extract_rgb_files = glob.glob(os.path.join(ood_extract_rgb_dir, oodSplit, "processed_rgb", "*", "*"))
    extracted_instances_list = [os.path.basename(file).split(".jpeg")[0] for file in ood_extract_rgb_files]

    auto_image_ids = []
    non_filtered_list = list(
        set(extracted_instances_list) - set(saved_instances_list) - set(deleted_instances_list) - set(auto_image_ids))

    print("Images to filter: ", len(non_filtered_list))
    count = 0
    while len(non_filtered_list) > 0:
        image_id = non_filtered_list[count]
        category = "_".join(image_id.split("_")[:-1])
        try:
            ood_image = cv2.imread(os.path.join(oodExtractFolder, oodSplit, "processed_rgb", category, image_id))
            cityscapes_mask_image = cv2.imread(os.path.join(oodExtractFolder, oodSplit, "cityscapes_mask", category, image_id.split(".")[0]+".png"), cv2.IMREAD_UNCHANGED)
            col_masks_image = cv2.imread(os.path.join(oodExtractFolder, oodSplit, "col_masks", category, image_id),  cv2.IMREAD_UNCHANGED)

            if ood_image is None or len(ood_image.shape) <= 2:
                count += 1

                deleted_id = non_filtered_list.pop(0)
                print("Image {} discarded".format((deleted_id)))
                with open(deleted_instance_list_file, 'a') as f:
                    f.write(deleted_id)
                    f.write("\n")

                continue

            # https://betterprogramming.pub/image-segmentation-python-7a838a464a84
            shape = ood_image.shape

            semantic_image = np.zeros((shape[0], shape[1]), dtype="uint8")
            panoptic_image = cityscapes_mask_image
            semantic_image[panoptic_image>0] = 50
            foreground_mask = np.zeros((shape[0], shape[1]), np.uint8)
            foreground_mask[panoptic_image>0] = 255
            instance_ids = []
            for ins in np.unique(np.unique(panoptic_image)):
                instance_ids.append(int(ins))
            instance_ids.remove(0)
            c = len(np.unique(panoptic_image))-1
            if str(c) in num_map:
                number_image = cv2.imread(num_map[str(c)])
            else:
                number_image = cv2.imread(num_map[str(10)])
            number_image = cv2.resize(number_image, (shape[1], shape[0]))

            horizontal = np.concatenate((ood_image, number_image, col_masks_image), axis=1)
            horizontal = cv2.resize(horizontal, (1024, 540))
            cv2.imshow(image_id+"real image, cityscapes mask, col masks ", horizontal)

            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    cv2.destroyAllWindows()
                    break
                elif k == 115 or k == 13:
                    # key "s" or "enter"
                    saved_id = non_filtered_list.pop(0)
                    print("Image {} saved with {} elements".format(saved_id, c))

                    to_copy_panoptic_file = os.path.join(saved_instance_dir, "ood_panoptic", image_id + ".png")
                    to_copy_rgb_file = os.path.join(saved_instance_dir, "ood_rgb", image_id + ".png")
                    to_copy_semantic_file = os.path.join(saved_instance_dir, "ood_semantic", image_id + ".png")

                    cv2.imwrite(to_copy_rgb_file, ood_image)
                    Image.fromarray(panoptic_image).save(to_copy_panoptic_file)
                    Image.fromarray(semantic_image).save(to_copy_semantic_file)

                    mask = np.where(foreground_mask > 0)
                    min_h = np.min(mask[0])
                    min_w = np.min(mask[1])
                    max_h = np.max(mask[0])
                    max_w = np.max(mask[1])

                    bbox = [int(min_h), int(min_w), int(max_h), int(max_w)]

                    category_name = saved_id.split("_")[0]

                    json_data = {"bbox": bbox,
                                 "instance_ids": instance_ids,
                                 "semantic_names": [category_name] * len(instance_ids),
                                 "rgb_image": os.path.join(oodSplit, "ood_rgb", image_id + ".png"),
                                 "panoptic_image": os.path.join(oodSplit, "ood_panoptic", image_id + ".png"),
                                 "semantic_image": os.path.join(oodSplit, "ood_semantic", image_id + ".png")}

                    saved_instance_gt[saved_id] = json_data

                    with open(saved_instance_list_file, 'a') as f:
                        f.write(saved_id)
                        f.write("\n")
                    with open(saved_instance_json_file, 'w') as f:
                        json.dump(saved_instance_gt, f, sort_keys=True, indent=4)

                    cv2.destroyAllWindows()
                    break
                elif k == 255 or k == 100:
                    # key "delete" or "d"
                    deleted_id = non_filtered_list.pop(0)
                    print("Image {} discarded with {} elements".format(deleted_id, c))
                    with open(deleted_instance_list_file, 'a') as f:
                        f.write(deleted_id)
                        f.write("\n")
                    cv2.destroyAllWindows()
                    break
        except:
            deleted_id = non_filtered_list.pop(0)
            print("Image {} discarded with {} elements".format(deleted_id, c))
            with open(deleted_instance_list_file, 'a') as f:
                f.write(deleted_id)
                f.write("\n")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood-extract-folder",
                        dest="oodExtractFolder",
                        help="path to the ood instances extracted folder",
                        default="/home/kumarasw/OOD_dataset/web_ood_rohit/web_scrap",
                        type=str)
    parser.add_argument("--filtered-path",
                        dest="filteredPath",
                        help="path to save filters instances",
                        default="/home/kumarasw/OOD_dataset/web_ood_rohit",
                        type=str)
    parser.add_argument("--ood-split",
                        dest="oodSplit",
                        help="Split for OOD objects. train or test",
                        default="train",
                        type=str)

    parser.add_argument("--use-file-to-add",
                        dest="useFileToAdd",
                        help="path to a file which contains list of comma separated image ids to save",
                        default="/home/kumarasw/OOD_dataset/save_files/filteres_coco/train/saved_files.txt",
                        type=str)

    parser.add_argument("--use-file-to-delete",
                        dest="useFileToDelete",
                        help="path to a file which contains list of comma separated image ids to delete",
                        default="",
                        type=str)

    args = parser.parse_args()

    display_to_filter_images(args)

# call the main
if __name__ == "__main__":
    main()