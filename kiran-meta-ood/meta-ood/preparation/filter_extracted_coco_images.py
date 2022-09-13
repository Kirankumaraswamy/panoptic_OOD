import cv2
import argparse
import os
import glob
import shutil
import json


def display_to_filter_images(args):
    cocoExtractFolder = args.cocoExtractFolder
    filteredPath = args.filteredPath
    oodSplit = args.oodSplit
    useFileToAdd = args.useFileToAdd
    useFileToDelete = args.useFileToDelete

    if not os.path.exists(cocoExtractFolder):
        raise "coco extract folder doesn't exist"

    if not os.path.exists(filteredPath):
        raise "Filter path doesn't exist"

    if oodSplit != "train" and oodSplit != "test":
        raise "COCO split value can be either train or test."

    coco_instance_files_dir = os.path.join(cocoExtractFolder, oodSplit)
    instance_json_file = os.path.join(cocoExtractFolder, oodSplit,  "coco_instances.json")
    instance_gt = json.load(open(instance_json_file))

    filter_parent_dir = "filtered_coco_instances"
    auto_image_ids = []
    auto_delete_ids = []

    # if user as already supplied the image_ids to be saved.
    if os.path.exists(useFileToAdd):
        with open(useFileToAdd, 'r') as f:
            lines = f.readlines()
        auto_image_ids = [line.split("\t")[0].strip() for line in lines]
    elif useFileToAdd != "":
        use_ids = useFileToAdd.split(",")
        use_ids = [use_id.strip() for use_id in use_ids]
        auto_image_ids = use_ids

    # if user as already supplied the image_ids to be saved.
    if os.path.exists(useFileToDelete):
        with open(useFileToDelete, 'r') as f:
            lines = f.readlines()
        auto_delete_ids = [line.split("\t")[0].strip() for line in lines]
    elif useFileToDelete != "":
        use_ids = useFileToDelete.split(",")
        use_ids = [use_id.strip() for use_id in use_ids]
        auto_delete_ids = use_ids

    saved_instance_dir = os.path.join(filteredPath, filter_parent_dir, oodSplit)
    saved_instance_json_file = os.path.join(filteredPath, filter_parent_dir, oodSplit, "coco_instances.json")

    if not os.path.exists(saved_instance_dir):
        os.makedirs(saved_instance_dir)
        os.makedirs(os.path.join(saved_instance_dir, "coco_panoptic"))
        os.makedirs(os.path.join(saved_instance_dir, "coco_rgb"))
        os.makedirs(os.path.join(saved_instance_dir, "coco_semantic"))

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
    coco_extract_rgb_dir = os.path.join(coco_instance_files_dir, "coco_rgb")
    cococ_extract_rgb_files = glob.glob(os.path.join(coco_extract_rgb_dir, "*"))
    extracted_instances_list = [os.path.basename(file).split("_rgb.png")[0] for file in cococ_extract_rgb_files]

    # if use file to add has any images in deleted image ids then reconsider them
    reconsider_ids = list(set(deleted_instances_list) & set(auto_image_ids))
    deleted_instances_list = list(set(deleted_instances_list) - set(reconsider_ids))

    print("Number of Image IDs already supplied to save: ", len(auto_image_ids))
    if len(auto_image_ids) > 0:
        for i in range(len(auto_image_ids)):
            image_id = auto_image_ids[i]
            if image_id not in  saved_instance_gt.keys():
                print("Image {} saved".format((image_id)))
                from_copy_panoptic_file = os.path.join(coco_instance_files_dir, "coco_panoptic", image_id + "_panoptic.png")
                from_copy_rgb_file = os.path.join(coco_instance_files_dir, "coco_rgb", image_id + "_rgb.png")
                from_copy_semantic_file = os.path.join(coco_instance_files_dir, "coco_semantic", image_id + "_semantic.png")

                to_copy_panoptic_file = os.path.join(saved_instance_dir, "coco_panoptic", image_id + "_panoptic.png")
                to_copy_rgb_file = os.path.join(saved_instance_dir, "coco_rgb", image_id + "_rgb.png")
                to_copy_semantic_file = os.path.join(saved_instance_dir, "coco_semantic", image_id + "_semantic.png")

                shutil.copyfile(from_copy_panoptic_file, to_copy_panoptic_file)
                shutil.copyfile(from_copy_rgb_file, to_copy_rgb_file)
                shutil.copyfile(from_copy_semantic_file, to_copy_semantic_file)

                saved_instance_gt[image_id] = instance_gt[image_id]

                with open(saved_instance_list_file, 'a') as f:
                    f.write(image_id)
                    f.write("\n")
                    saved_instances_list.append(image_id)
                with open(saved_instance_json_file, 'w') as f:
                    json.dump(saved_instance_gt, f, sort_keys=True, indent=4)


        # save the modified deleted list file again
        with open(deleted_instance_list_file, 'w') as f:
            for i, id in enumerate(deleted_instances_list):
                f.write(id)
                f.write("\n")

    # if use file to delete has amy images in saved image ids then remove them
    remove_ids = list(set(saved_instances_list) & set(auto_delete_ids))
    saved_instances_list = list(set(saved_instances_list) - set(remove_ids))
    print("Number of Image IDs supplied to delete from already saved images: ", len(auto_delete_ids))

    if len(auto_delete_ids) > 0:
        for i in range(len(auto_delete_ids)):
            image_id = auto_delete_ids[i]
            if image_id in saved_instance_gt.keys():
                print("Image {} deleting..".format((image_id)))

                delete_panoptic_file = os.path.join(saved_instance_dir, "coco_panoptic", image_id + "_panoptic.png")
                delete_rgb_file = os.path.join(saved_instance_dir, "coco_rgb", image_id + "_rgb.png")
                delete_semantic_file = os.path.join(saved_instance_dir, "coco_semantic", image_id + "_semantic.png")

                if os.path.exists(delete_panoptic_file):
                    os.remove(delete_panoptic_file)
                    os.remove(delete_rgb_file)
                    os.remove(delete_semantic_file)

                saved_instance_gt.pop(image_id)

                with open(deleted_instance_list_file, 'a') as f:
                    f.write(image_id)
                    f.write("\n")
                    deleted_instances_list.append(image_id)
        with open(saved_instance_json_file, 'w') as f:
            json.dump(saved_instance_gt, f, sort_keys=True, indent=4)


        # save the modified deleted list file again
        with open(saved_instance_list_file, 'w') as f:
            for i, id in enumerate(saved_instances_list):
                f.write(id)
                f.write("\n")

    # get IDs of instances to be filtered out
    non_filtered_list = list(
    set(extracted_instances_list) - set(saved_instances_list) - set(deleted_instances_list) - set(auto_image_ids))
    print("Images to filter: ", len(non_filtered_list))
    while len(non_filtered_list) > 0:
        image_id = non_filtered_list[0]
        img = cv2.imread(os.path.join(coco_instance_files_dir, "coco_rgb", image_id+"_rgb.png"))
        cv2.imshow(image_id, img)
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                cv2.destroyAllWindows()
                break
            elif k == 115 or k == 13:
                # key "s" or "enter"
                saved_id = non_filtered_list.pop(0)
                print("Image {} saved".format((saved_id)))
                from_copy_panoptic_file = os.path.join(coco_instance_files_dir, "coco_panoptic", image_id+"_panoptic.png")
                from_copy_rgb_file = os.path.join(coco_instance_files_dir, "coco_rgb", image_id + "_rgb.png")
                from_copy_semantic_file = os.path.join(coco_instance_files_dir, "coco_semantic", image_id + "_semantic.png")

                to_copy_panoptic_file = os.path.join(saved_instance_dir, "coco_panoptic", image_id + "_panoptic.png")
                to_copy_rgb_file = os.path.join(saved_instance_dir, "coco_rgb", image_id + "_rgb.png")
                to_copy_semantic_file = os.path.join(saved_instance_dir, "coco_semantic", image_id + "_semantic.png")

                shutil.copyfile(from_copy_panoptic_file, to_copy_panoptic_file)
                shutil.copyfile(from_copy_rgb_file, to_copy_rgb_file)
                shutil.copyfile(from_copy_semantic_file, to_copy_semantic_file)

                saved_instance_gt[saved_id] = instance_gt[saved_id]

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
                print("Image {} discarded".format((deleted_id)))
                with open(deleted_instance_list_file, 'a') as f:
                    f.write(deleted_id)
                    f.write("\n")
                cv2.destroyAllWindows()
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-extract-folder",
                        dest="cocoExtractFolder",
                        help="path to the COCO instances extracted folder",
                        default="/home/kumarasw/OOD_dataset/coco_instances",
                        type=str)
    parser.add_argument("--filtered-path",
                        dest="filteredPath",
                        help="path to save filters instances",
                        default="/home/kumarasw/OOD_dataset",
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

