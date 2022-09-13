import argparse
from panoptic_evaluation.cityscapes_ood import CityscapesOOD
import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import glob

from config import config_training_setup
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
from cityscapesscripts.helpers.labels import name2label

working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
instance_working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_instance_eval_")

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


def semantic_process(inputs, outputs):
    for i in range(len(outputs)):
        input = inputs[i]
        output = outputs[i]
        file_name = input["file_name"]
        basename = os.path.splitext(os.path.basename(file_name))[0]
        pred_filename = os.path.join(working_dir.name, basename + "_pred.png")

        #output["sem_seg"].argmax(dim=0).cpu().numpy()
        output = output["sem_seg"].numpy()
        pred = 255 * np.ones(output.shape, dtype=np.uint8)
        for train_id, label in trainId2label.items():
            if label.ignoreInEval:
                continue
            pred[output == train_id] = label.id
        Image.fromarray(pred).save(pred_filename)
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

def sematic_evaluate(gt_data_path):
    # Load the Cityscapes eval script *after* setting the required env var,
    # since the script reads CITYSCAPES_DATASET into global variables at load time.
    import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

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
        file = cityscapes_eval.getPrediction(cityscapes_eval.args, gt)
        if file is not None:
            predictionImgList.append(file)
            gtImgList.append(gt)
    results = cityscapes_eval.evaluateImgLists(
        predictionImgList, gtImgList, cityscapes_eval.args
    )
    ret = OrderedDict()
    ret["sem_seg"] = {
        "IoU": 100.0 * results["averageScoreClasses"],
        "iIoU": 100.0 * results["averageScoreInstClasses"],
        "IoU_sup": 100.0 * results["averageScoreCategories"],
        "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
    }
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
        file = cityscapes_instance_eval.getPrediction(gt, cityscapes_instance_eval.args)
        if file is not None:
            predictionImgList.append(file)
            gtImgList.append(gt)
    results = cityscapes_instance_eval.evaluateImgLists(
        predictionImgList, gtImgList, cityscapes_instance_eval.args
    )["averages"]

    ret = OrderedDict()
    ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
    return ret

def panoptic_process(inputs, outputs):
    predictions = []
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

        del output
    del outputs
    torch.cuda.empty_cache()
    return predictions

def panoptic_evaluate(predictions, gt_json_path, gt_data_path):

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



def data_load(root=None, split="val", transform=None):
    datset = CityscapesOOD(root, split, transform )
    return datset

def data_evaluate(estimator=None, evaluation_dataset=None, batch_size=1, collate_fn=None, semantic_only=False):

    dataloader = DataLoader(evaluation_dataset, batch_size=batch_size,
                    collate_fn=collate_fn)
    predictions = []
    for count, inputs in enumerate(dataloader):
        print("count: ", count)
        logits = estimator(inputs)

        semantic_process(inputs, logits)
        if not semantic_only:
            predictions += panoptic_process(inputs, logits)
            instance_process(inputs, logits)
        del logits
        torch.cuda.empty_cache()

    gt_path = evaluation_dataset.root
    semantic_result = sematic_evaluate(os.path.join(gt_path, "gtFine", evaluation_dataset.split))
    print(semantic_result)

    if not semantic_only:
        panoptic_result = panoptic_evaluate(predictions, os.path.join(gt_path, "gtFine", "cityscapes_panoptic_"+evaluation_dataset.split +".json"), os.path.join(gt_path, "gtFine", "cityscapes_panoptic_"+evaluation_dataset.split))
        instance_result = instance_evaluate(os.path.join(gt_path, "gtFine", evaluation_dataset.split))

    working_dir.cleanup()
    instance_working_dir.cleanup()

    result = {}
    result["semantic_seg"] = semantic_result

    if not semantic_only:
        result["panotic_seg"] = panoptic_result
        result["instance_seg"] = instance_result
    return  result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-root", "--ROOT", nargs="?", type=str)
    parser.add_argument("-split", "--SPLIT", nargs="?", type=str)

    args = vars(parser.parse_args())
    data_evaluate(args)
