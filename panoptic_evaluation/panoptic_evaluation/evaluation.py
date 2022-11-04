import argparse
import math

from panoptic_evaluation.cityscapes_ood import CityscapesOOD
import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import glob

from torch.utils.data import DataLoader
from panopticapi.utils import id2rgb

from PIL import Image
import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval
import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_instance_eval
from cityscapesscripts.helpers.labels import id2label, labels, trainId2label
import io
import contextlib
import itertools
import json
import tempfile
from collections import OrderedDict
from tabulate import tabulate
import fnmatch
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, \
    precision_score, recall_score

from cityscapesscripts.helpers.labels import name2label, id2label, trainId2label, category2labels

# Load the Cityscapes eval script *after* setting the required env var,
# since the script reads CITYSCAPES_DATASET into global variables at load time.

import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval
import cityscapesscripts.helpers.labels as cityscapes_labels

working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
instance_working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_instance_eval_")
anomaly_working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_anomaly_eval_")

cityscapes_categories = []
useTrainId = False
thing_dataset_id_to_contiguous_id = {}
train_id_to_id = {}
id_to_train_id = {}
train_id_to_name = {}
for label in labels:
    train_id_to_id[label.trainId] = label.id
    id_to_train_id[label.id] = label.trainId
    train_id_to_name[label.trainId] = label.name
    if label.ignoreInEval:
        continue
    isthing = 1 if label.hasInstances else 0
    if isthing:
        thing_dataset_id_to_contiguous_id[int(label.id)] = int(label.trainId)
    cityscapes_categories.append({'id': int(label.trainId) if useTrainId else int(label.id),
                                  'name': label.name,
                                  'color': label.color,
                                  'supercategory': label.category,
                                  'isthing': isthing})


def semantic_process(inputs, outputs, evaluate_ood):
    for i in range(len(outputs)):
        input = inputs[i]
        out = outputs[i]
        file_name = input["file_name"]
        basename = os.path.splitext(os.path.basename(file_name))[0]
        pred_filename = os.path.join(working_dir.name, basename + "_pred.png")

        # output["sem_seg"].argmax(dim=0).cpu().numpy()
        output = out["sem_seg"].numpy()

        # zero is unlabelled in non training IDs
        pred = np.zeros(output.shape, dtype=np.uint8)

        for train_id, label in trainId2label.items():
            if label.ignoreInEval:
                continue
            pred[output == train_id] = label.id
        Image.fromarray(pred).save(pred_filename)

        if evaluate_ood:
            pred_filename_ood = os.path.join(working_dir.name, basename + "_pred_ood.png")
            output_ood = out["sem_seg_ood"].numpy()
            pred_ood = np.zeros(output_ood.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval and not label.name == "OOD":
                    continue
                pred_ood[output_ood == train_id] = label.id
            Image.fromarray(pred_ood).save(pred_filename_ood)
        del output
    del outputs
    torch.cuda.empty_cache()


def anomaly_process(inputs, outputs):
    for i in range(len(outputs)):
        input = inputs[i]
        output = outputs[i]
        file_name = input["file_name"]
        basename = os.path.splitext(os.path.basename(file_name))[0]
        pred_filename = os.path.join(anomaly_working_dir.name, basename + "_pred")

        # output["sem_seg"].argmax(dim=0).cpu().numpy()
        output = output["anomaly_score"].numpy()
        np.save(pred_filename, output)
        del output
    del outputs
    torch.cuda.empty_cache()


def instance_process(inputs, outputs):
    for i in range(len(outputs)):
        input = inputs[i]
        output = outputs[i]
        file_name = input["file_name"]
        basename = os.path.splitext(os.path.basename(file_name))[0]
        pred_txt = os.path.join(instance_working_dir.name, basename + "_pred.txt")

        if "instances" in output:
            output = output["instances"]
            num_instances = len(output)
            with open(pred_txt, "w") as fout:
                for i in range(num_instances):
                    pred_class = output.pred_classes[i].cpu()
                    classes = train_id_to_name[pred_class.item()]
                    class_id = name2label[classes].id
                    score = output.scores[i].cpu()
                    mask = output.pred_masks[i].cpu().numpy().astype("uint8")
                    png_filename = os.path.join(
                        instance_working_dir.name, basename + "_{}_{}.png".format(i, classes)
                    )

                    Image.fromarray(mask * 255).save(png_filename)
                    fout.write(
                        "{} {} {}\n".format(os.path.basename(png_filename), class_id, score)
                    )
        else:
            # Cityscapes requires a prediction file for every ground truth image.
            with open(pred_txt, "w") as fout:
                pass
        del output
    del outputs
    torch.cuda.empty_cache()


def sematic_evaluate(gt_data_path, evaluate_ood):
    print("Evaluating results under {} ...".format(working_dir.name))

    # set some global states in cityscapes evaluation API, before evaluating
    cityscapes_eval.args.predictionPath = os.path.abspath(working_dir.name)
    cityscapes_eval.args.predictionWalk = None
    cityscapes_eval.args.JSONOutput = False
    cityscapes_eval.args.colorized = False

    # These lines are adopted from
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
    gt_dir = gt_data_path
    groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
    assert len(
        groundTruthImgList
    ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
        cityscapes_eval.args.groundTruthSearch
    )
    predictionImgList = []
    gtImgList = []
    for gt in groundTruthImgList:
        try:
            file = cityscapes_eval.getPrediction(cityscapes_eval.args, gt)
        except:
            file = None
        if file is not None:
            predictionImgList.append(file)
            gtImgList.append(gt)
    results = cityscapes_eval.evaluateImgLists(
        predictionImgList, gtImgList, cityscapes_eval.args
    )

    if evaluate_ood:

        gt_list = []
        pred_list = []
        ood_pred_list = []
        # Evaluate all pairs of images and save them into a matrix
        for i in range(len(predictionImgList)):
            predictionImgFileName = predictionImgList[i]
            groundTruthImgFileName = gtImgList[i]
            predictionImgOODFileName = predictionImgList[i].split("_pred")[0] + "_pred_ood" + \
                                       predictionImgList[i].split("_pred")[1]

            try:
                predictionImg = Image.open(predictionImgFileName)
                predictionNp = np.array(predictionImg)
            except:
                printError("Unable to load " + predictionImgFileName)
            try:
                groundTruthImg = Image.open(groundTruthImgFileName)
                groundTruthNp = np.array(groundTruthImg)
            except:
                printError("Unable to load " + groundTruthImgFileName)
            try:
                predictionImgOOD = Image.open(predictionImgOODFileName)
                predictionNpOOD = np.array(predictionImgOOD)
            except:
                printError("Unable to load " + predictionImgOODFileName)

            gt_list.append(np.expand_dims(groundTruthNp, 0))
            pred_list.append(np.expand_dims(predictionNp, 0))
            ood_pred_list.append(np.expand_dims(predictionNpOOD, 0))

        gts = np.array(gt_list)
        preds = np.array(pred_list)
        ood_preds = np.array(ood_pred_list)

        mask = 255 * np.ones(gts.shape, dtype=np.uint8)
        in_pixel_count = 0
        for id, label in id2label.items():
            if label.ignoreInEval and not label.name == "OOD":
                continue
            if label.name == "OOD":
                mask[gts == id] = 1
            else:
                # consider only True positives from the model prediction
                mask[np.logical_and(gts == id, preds == id)] = 0
                in_pixel_count += np.sum(gts == id)

        print("Total in-distribution pixels for evaluation: ", in_pixel_count)
        print("Total no of in-distribution TP pixels considered: ", np.sum(mask == 0))
        print("Total no of OOD pixels considered: ", np.sum(mask == 1))

        ood_pred_in = ood_preds[np.where(mask == 0)]
        ood_pred_out = ood_preds[np.where(mask == 1)]

        in_pred = np.zeros(len(ood_pred_in))
        out_pred = np.ones(len(ood_pred_out))

        in_pred[np.where(ood_pred_in == 50)] = 1
        out_pred[np.where(ood_pred_out != 50)] = 0

        in_gt = np.zeros(len(in_pred))
        out_gt = np.ones(len(out_pred))

        TP = np.sum(out_pred == 1)
        FP = np.sum(in_pred == 1)
        FN = np.sum(out_pred == 0)
        TN = np.sum(in_pred == 0)

        print("Total number of in distribution pixels identified correctly: ", TN)
        print("Total number of Out distribution pixels identified correctly: ", TP)

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1score = 2 * (precision * recall) / (precision + recall)

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        gmean = math.sqrt(sensitivity * specificity)

        performance_without_uncertainty = len(ood_pred_in) / (len(ood_pred_in) + len(out_pred))
        performance_with_uncertainty = (TP + TN) / (len(ood_pred_in) + len(out_pred))

        print("------------------------------------------")
        print("          OOD semantic results ")
        print("------------------------------------------")
        print("UF1                              : ", f1score)
        print("UPrecision                       : ", precision)
        print("URecall                          : ", recall)
        print("USensitivity                     : ", sensitivity)
        print("USpecificity                     : ", specificity)
        print("UGmean                           : ", gmean)
        print("Performance with Uncertainity    : ", performance_with_uncertainty)
        print("Performance without Uncertainity : ", performance_without_uncertainty)
        print("------------------------------------------")


    ret = OrderedDict()
    ret["sem_seg"] = {
        "IoU": 100.0 * results["averageScoreClasses"],
        "iIoU": 100.0 * results["averageScoreInstClasses"],
        "IoU_sup": 100.0 * results["averageScoreCategories"],
        "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
    }
    if evaluate_ood:
        ret["sem_seg"]["uF1"] = f1score
        ret["sem_seg"]["uPrecision"] = precision
        ret["sem_seg"]["uRecall"] = recall
        ret["sem_seg"]["uSensitivity"] = sensitivity
        ret["sem_seg"]["uSpecificity"] = specificity
        ret["sem_seg"]["uGmean"] = gmean
        ret["sem_seg"]["performance_with_uncertainity"] = performance_with_uncertainty
        ret["sem_seg"]["performance_without_uncertainity"] = performance_without_uncertainty
    return ret


def instance_evaluate(gt_data_path):
    print("Evaluating results under {} ...".format(instance_working_dir.name))

    # set some global states in cityscapes evaluation API, before evaluating
    cityscapes_instance_eval.args.predictionPath = os.path.abspath(instance_working_dir.name)
    cityscapes_instance_eval.args.predictionWalk = None
    cityscapes_instance_eval.args.JSONOutput = False
    cityscapes_instance_eval.args.colorized = False
    cityscapes_instance_eval.args.gtInstancesFile = os.path.join(instance_working_dir.name, "gtInstances.json")

    gt_dir = gt_data_path
    groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_instanceIds.png"))
    assert len(
        groundTruthImgList
    ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
        cityscapes_instance_eval.args.groundTruthSearch
    )
    predictionImgList = []
    gtImgList = []
    for gt in groundTruthImgList:
        try:
            file = cityscapes_instance_eval.getPrediction(gt, cityscapes_instance_eval.args)
        except:
            file = None
        if file is not None:
            predictionImgList.append(file)
            gtImgList.append(gt)
    results = cityscapes_instance_eval.evaluateImgLists(
        predictionImgList, gtImgList, cityscapes_instance_eval.args
    )["averages"]

    ret = OrderedDict()
    ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
    return ret


def anomaly_evaluate(gt_data_path):
    print("Evaluating anomaly results under {} ...".format(anomaly_working_dir.name))

    # set some global states in cityscapes evaluation API, before evaluating
    predictionPath = os.path.abspath(anomaly_working_dir.name)
    predictionWalk = None

    gt_dir = gt_data_path
    groundTruthImgList = glob.glob(os.path.join(gt_dir, "*", "*_gtFine_labelIds.png"))
    assert len(
        groundTruthImgList
    ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
        cityscapes_eval.args.groundTruthSearch
    )

    # walk the prediction path, if not happened yet
    if not predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(predictionPath):
            walk.append((root, filenames))
        predictionWalk = walk

    ood_gts_list = []
    anomaly_score_list = []
    for gt in groundTruthImgList:
        predictionFile = None
        pattern = os.path.basename(gt).split("_")
        city_name = pattern[0]
        sequence_nb = pattern[1]
        frame_nb = pattern[2]
        filePattern = "{}_{}_{}*.npy".format(city_name, sequence_nb, frame_nb)
        for root, filenames in predictionWalk:
            for filename in fnmatch.filter(filenames, filePattern):
                if not predictionFile:
                    predictionFile = os.path.join(root, filename)
                else:
                    print("Found multiple predictions for ground truth {}".format(gt))

        if predictionFile is not None:
            anomaly_score = np.load(predictionFile)
            anomaly_score = np.array(anomaly_score)
            ground = Image.open(gt)
            ood_gts = np.array(ground)
            ood_gts_list.append(np.expand_dims(ood_gts, 0))
            anomaly_score_list.append(np.expand_dims(anomaly_score, 0))

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    mask = 255 * np.ones(ood_gts.shape, dtype=np.uint8)
    for id, label in id2label.items():
        if label.ignoreInEval and not label.name == "OOD":
            continue
        if label.name == "OOD":
            mask[ood_gts == id] = 1
        else:
            mask[ood_gts == id] = 0

    # drop void pixels
    ood_mask = (mask == 1)
    ind_mask = (mask == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    print('Measuring anomaly metrics...')
    ood_weight = len(ood_out) / (len(ood_out) + len(ind_out))
    ind_weight = len(ind_out) / (len(ood_out) + len(ind_out))

    fpr, tpr, _ = roc_curve(val_label, val_out)

    min_fpr = 100
    tp = 0
    th = 0
    max_th = -99999
    max_score = -99999
    max_tpr = -99999
    max_fpr = -99999
    max_score = -99999
    max_score_th = -99999
    max_score_fpr = -99999
    max_score_tpr = -99999

    for i in range(len(fpr)):
        if tpr[i] >= 0.95 and fpr[i] < min_fpr:
            min_fpr = fpr[i]
            tp = tpr[i]
            th = _[i]
        if fpr[i] <= 0.05 and tpr[i] > max_tpr:
            max_tpr = tpr[i]
            max_th = _[i]
            max_fpr = fpr[i]
        score = tpr[i] / fpr[i]
        if score > max_score:
            max_score = score
            max_score_th = _[i]
            max_score_fpr = fpr[i]
            max_score_tpr = tpr[i]

    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(val_label, val_out)
    prc_auc = average_precision_score(val_label, val_out)

    max_area = -1
    max_threshold = -99999
    max_precision = -1
    max_recall = -1
    for i in range(len(precision)):
        area = precision[i] * recall[i]
        if area > max_area:
            max_area = area
            max_threshold = _[i]
            max_precision = precision[i]
            max_recall = recall[i]

    return {
        "FPR@95%TPR": min_fpr,
        "TPR@5%FPR": max_tpr,
        "Threshold@95%TPR": th,
        "Threshold@5%FPR": max_th,
        "AP": prc_auc,
        "auroc": roc_auc,
        "PRthreshold": max_threshold,
        "TPRvsFPRThreshold": max_score_th
    }


def panoptic_process(inputs, outputs, evaluate_ood):
    predictions = []
    predictions_OOD = []
    for i in range(len(outputs)):
        input = inputs[i]
        output = outputs[i]
        panoptic_img, segments_info = output["panoptic_seg"]
        panoptic_img = panoptic_img.cpu().numpy()

        if segments_info is None:
            # If "segments_info" is None, we assume "panoptic_img" is a
            # H*W int32 image storing the panoptic_id in the format of
            # category_id * label_divisor + instance_id. We reserve -1 for
            # VOID label, and add 1 to panoptic_img since the official
            # evaluation script uses 0 for VOID label.
            label_divisor = 1000
            segments_info = []
            for panoptic_label in np.unique(panoptic_img):
                if panoptic_label == -1:
                    # VOID region.
                    continue
                pred_class = panoptic_label // label_divisor
                isthing = (
                        pred_class in thing_dataset_id_to_contiguous_id.values()
                )
                segments_info.append(
                    {
                        "id": int(panoptic_label) + 1,
                        "category_id": train_id_to_id[int(pred_class)],
                        "isthing": bool(isthing),
                    }
                )
            # Official evaluation script uses 0 for VOID label.
            panoptic_img += 1

        file_name = os.path.basename(input["file_name"])
        file_name_png = os.path.splitext(file_name)[0] + ".png"
        with io.BytesIO() as out:
            Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
            predictions.append({
                "image_id": input["image_id"],
                "file_name": file_name_png,
                "png_string": out.getvalue(),
                "segments_info": segments_info,
            })

        if evaluate_ood:
            panoptic_img_ood, segments_info_ood = output["panoptic_seg_ood"]
            panoptic_img_ood = panoptic_img_ood.cpu().numpy()
            # for OOD prediction
            if segments_info_ood is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = 1000
                segments_info_ood = []
                for panoptic_label in np.unique(panoptic_img_ood):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                            pred_class in thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info_ood.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": train_id_to_id[int(pred_class)],
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img_ood += 1

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + "_ood" + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img_ood)).save(out, format="PNG")
                predictions_OOD.append({
                    "image_id": input["image_id"],
                    "file_name": file_name_png,
                    "png_string": out.getvalue(),
                    "segments_info": segments_info_ood,
                })

        del output
    del outputs
    torch.cuda.empty_cache()
    return predictions, predictions_OOD


def panoptic_evaluate(predictions, predictions_ood, gt_json_path, gt_data_path, evaluate_ood):
    # PanopticApi requires local files
    gt_json = gt_json_path
    gt_folder = gt_data_path

    with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
        print("Writing all panoptic predictions to {} ...".format(pred_dir))
        for p in predictions:
            with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                f.write(p.pop("png_string"))

        with open(gt_json, "r") as f:
            json_data = json.load(f)
        json_data["annotations"] = predictions

        if evaluate_ood:
            # writing ood data
            for p in predictions_ood:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))
            json_data["annotations_ood"] = predictions_ood

        output_dir = pred_dir
        predictions_json = os.path.join(output_dir, "predictions.json")
        with open(predictions_json, "w") as f:
            f.write(json.dumps(json_data))

        from panopticapi.evaluation import pq_compute

        with contextlib.redirect_stdout(io.StringIO()):
            pq_res = pq_compute(
                gt_json,
                predictions_json,
                gt_folder=gt_folder,
                pred_folder=pred_dir,
                evaluate_ood=evaluate_ood
            )

    res = {}
    res["PQ"] = 100 * pq_res["All"]["pq"]
    res["SQ"] = 100 * pq_res["All"]["sq"]
    res["RQ"] = 100 * pq_res["All"]["rq"]
    res["PQ_th"] = 100 * pq_res["Things"]["pq"]
    res["SQ_th"] = 100 * pq_res["Things"]["sq"]
    res["RQ_th"] = 100 * pq_res["Things"]["rq"]
    res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
    res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
    res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

    if evaluate_ood:
        res["UPQ"] = 100 * pq_res["OOD"][0]["upq"]
        res["USQ"] = 100 * pq_res["OOD"][0]["usq"]
        res["URQ"] = 100 * pq_res["OOD"][0]["urq"]
        res["UPQ_in"] = 100 * pq_res["OOD"][1][0]["upq"]
        res["USQ_in"] = 100 * pq_res["OOD"][1][0]["usq"]
        res["URQ_in"] = 100 * pq_res["OOD"][1][0]["urq"]
        res["UPQ_out"] = 100 * pq_res["OOD"][1][1]["upq"]
        res["USQ_out"] = 100 * pq_res["OOD"][1][1]["usq"]
        res["URQ_out"] = 100 * pq_res["OOD"][1][1]["urq"]

    results = OrderedDict({"panoptic_seg": res})
    print_panoptic_results(pq_res)
    return results


def print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    print("Panoptic Evaluation Results:\n" + table)

    if "OOD" in pq_res:
        headers = ["", "UPQ", "USQ", "URQ", "#categories"]
        ood_data = []
        ood_data.append(["All"] + [pq_res["OOD"][0][k] * 100 for k in ["upq", "usq", "urq"]] + [2])
        ood_data.append(["IN_PIXELS"] + [pq_res["OOD"][1][0][k] * 100 for k in ["upq", "usq", "urq"]] + [1])
        ood_data.append(["OUT_PIXELS"] + [pq_res["OOD"][1][1][k] * 100 for k in ["upq", "usq", "urq"]] + [1])
        ood_table = tabulate(
            ood_data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
                             )
        print("OOD Panoptic Evaluation Results:\n" + ood_table)

        print("Total number of in-distribution instances: ", pq_res["OOD"][1][0]["no_instances"])
        print("Number of in-distribution instances correctly identified (true positives): ", pq_res["OOD"][1][0]["correct_instances"])
        print("Number of in-distribution instances wrongly identified (false positives): ", pq_res["OOD"][1][0]["false_instances"])
        print("Total number of out-distribution instances: ", pq_res["OOD"][1][1]["no_instances"])
        print("Number of out-distribution instances correctly identified (true positives): ", pq_res["OOD"][1][1]["correct_instances"])
        print("Number of out-distribution instances wrongly identified (false positives): ", pq_res["OOD"][1][1]["false_instances"])


def data_load(root=None, split="val", transform=None)
    datset = CityscapesOOD(root, split, transform)
    return datset


def data_evaluate(estimator=None, evaluation_dataset=None, batch_size=1, collate_fn=None, evaluate_ood=True, semantic_only=False):
    dataloader = DataLoader(evaluation_dataset, batch_size=batch_size,
                            collate_fn=collate_fn)
    predictions = []
    predictions_ood = []
    has_anomoly = False
    for count, inputs in enumerate(dataloader):
        print("count: ", count)
        logits = estimator(inputs)

        semantic_process(inputs, logits, evaluate_ood)
        if evaluate_ood and "anomaly_score" in logits[0].keys():
            anomaly_process(inputs, logits)
            has_anomoly = True

        if not semantic_only:
            pred, pred_ood = panoptic_process(inputs, logits, evaluate_ood)
            predictions += pred
            predictions_ood += pred_ood
            instance_process(inputs, logits)
        del logits
        torch.cuda.empty_cache()

        if count == 2:
            break

    gt_path = evaluation_dataset.root
    result = {}

    semantic_result = sematic_evaluate(os.path.join(gt_path, "gtFine", evaluation_dataset.split), evaluate_ood)
    result["semantic_seg"] = semantic_result

    if not semantic_only:
        panoptic_result = panoptic_evaluate(predictions, predictions_ood, os.path.join(gt_path, "gtFine",
                                                                                       "cityscapes_panoptic_" + evaluation_dataset.split + ".json"),
                                            os.path.join(gt_path, "gtFine",
                                                         "cityscapes_panoptic_" + evaluation_dataset.split), evaluate_ood)
        instance_result = instance_evaluate(os.path.join(gt_path, "gtFine", evaluation_dataset.split))

        result["panotic_seg"] = panoptic_result
        result["instance_seg"] = instance_result

    if evaluate_ood and has_anomoly:
        anomoly_result = anomaly_evaluate(os.path.join(gt_path, "gtFine", evaluation_dataset.split))
        result["anomoly_result"] = anomoly_result

    working_dir.cleanup()
    instance_working_dir.cleanup()
    anomaly_working_dir.cleanup()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-root", "--ROOT", nargs="?", type=str)
    parser.add_argument("-split", "--SPLIT", nargs="?", type=str)

    args = vars(parser.parse_args())
    data_evaluate(args)
