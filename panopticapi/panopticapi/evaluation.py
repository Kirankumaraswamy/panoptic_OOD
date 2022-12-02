#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import time
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing

import PIL.Image as Image

from panopticapi.utils import get_traceback, rgb2id

OFFSET = 256 * 256 * 256
VOID = 0


class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


class UPQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, upq_stat_cat):
        self.iou += upq_stat_cat.iou
        self.tp += upq_stat_cat.tp
        self.fp += upq_stat_cat.fp
        self.fn += upq_stat_cat.fn
        return self


class UPQStat():
    def __init__(self):
        self.upq_per_cat = defaultdict(UPQStatCat)

    def __getitem__(self, i):
        return self.upq_per_cat[i]

    def __iadd__(self, upq_stat):
        for label, upq_stat_cat in upq_stat.upq_per_cat.items():
            self.upq_per_cat[label] += upq_stat_cat
        return self

    def upq_average(self):
        upq, usq, urq, n = 0, 0, 0, 0
        per_class_results = {}
        for i in range(2):
            iou = self.upq_per_cat[i].iou
            tp = self.upq_per_cat[i].tp
            fp = self.upq_per_cat[i].fp
            fn = self.upq_per_cat[i].fn
            if tp + fp + fn == 0:
                per_class_results[i] = {'upq': 0.0, 'usq': 0.0, 'urq': 0.0}
                continue
            n += 1
            upq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            usq_class = iou / tp if tp != 0 else 0
            urq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[i] = {'upq': upq_class, 'usq': usq_class, 'urq': urq_class, 'no_instances': (tp+fn), "correct_instances": tp, "false_instances": fp}
            upq += upq_class
            usq += usq_class
            urq += urq_class

        return {'upq': upq / n, 'usq': usq / n, 'urq': urq / n, 'n': n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories, evaluate_ood):
    pq_stat = PQStat()
    upq_stat = UPQStat()

    idx = 0
    for gt_ann, pred_ann, pred_ann_ood in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError(
                    'In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(
                        gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                raise KeyError(
                    'In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'],
                                                                                                    label,
                                                                                                    pred_segms[label][
                                                                                                        'category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError(
                'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(
                    gt_ann['image_id'], list(pred_labels_set)))


        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        matched_catagory_id = []
        matched_ious = {}
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue
            # ignore OOD segments
            if gt_segms[gt_label]['category_id'] == 50:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                (VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
                matched_catagory_id.append(gt_segms[gt_label]['category_id'])
                matched_ious[gt_label] = iou

        # count false negatives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            # ignore OOD
            if gt_segms[gt_label]['category_id'] == 50:
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        pred_fp_label_id_map = {}
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # ignore OOD
            if pred_info['category_id'] == 50:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
            pred_fp_label_id_map[pred_label] = pred_info['category_id']


        if pred_ann_ood is not None:
            pan_pred_ood = np.array(Image.open(os.path.join(pred_folder, pred_ann_ood['file_name'])), dtype=np.uint32)
            pan_pred_ood = rgb2id(pan_pred_ood)
            pred_segms_ood = {el['id']: el for el in pred_ann_ood['segments_info']}

            # predicted segments with OOD area calculation + prediction sanity checks
            pred_labels_set_ood = set(el['id'] for el in pred_ann_ood['segments_info'])
            labels_ood, labels_cnt_ood = np.unique(pan_pred_ood, return_counts=True)
            for label_ood, label_cnt_ood in zip(labels_ood, labels_cnt_ood):
                if label_ood not in pred_segms_ood:
                    if label_ood == VOID:
                        continue
                    raise KeyError(
                        'In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(
                            gt_ann['image_id'], label_ood))
                pred_segms_ood[label_ood]['area'] = label_cnt_ood
                pred_labels_set_ood.remove(label_ood)
                if pred_segms_ood[label_ood]['category_id'] not in categories:
                    raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(
                        gt_ann['image_id'], label_ood, pred_segms_ood[label_ood]['category_id']))
            if len(pred_labels_set_ood) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(
                        gt_ann['image_id'], list(pred_labels_set_ood)))

            # for OOD panoptic prediction
            # confusion matrix calculation
            pan_gt_pred_ood = pan_gt.astype(np.uint64) * OFFSET + pan_pred_ood.astype(np.uint64)
            gt_pred_map_ood = {}
            labels_ood, labels_cnt_ood = np.unique(pan_gt_pred_ood, return_counts=True)
            for label_ood, intersection in zip(labels_ood, labels_cnt_ood):
                gt_id = label_ood // OFFSET
                pred_id = label_ood % OFFSET
                gt_pred_map_ood[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_matched_ood = set()
            pred_matched_ood = set()
            matched_catagory_id_ood = []
            for label_tuple, intersection in gt_pred_map_ood.items():
                gt_label, pred_label_ood = label_tuple
                if gt_label not in gt_segms:
                    continue
                if pred_label_ood not in pred_segms_ood:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms_ood[pred_label_ood]['category_id']:
                    continue

                union = pred_segms_ood[pred_label_ood]['area'] + gt_segms[gt_label][
                    'area'] - intersection - gt_pred_map_ood.get(
                    (VOID, pred_label_ood), 0)
                iou = intersection / union
                if iou > 0.5 and gt_label in matched_ious:
                    gt_matched_ood.add(gt_label)
                    pred_matched_ood.add(pred_label_ood)
                    matched_catagory_id_ood.append(gt_segms[gt_label]['category_id'])

                    if not gt_segms[gt_label]['category_id'] == 50:
                        # calculate the amount of pixels overlaping with the prediction before applying uncertainity
                        non_ood_iou = matched_ious[gt_label]
                        ood_iou = iou / non_ood_iou
                        upq_stat[0].tp += 1
                        upq_stat[0].iou += ood_iou
                    else:
                        upq_stat[1].tp += 1
                        upq_stat[1].iou += iou

            # count false negatives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched_ood:
                    continue
                # we should not consider for uncertainity estimation of false negatives if the model fails to identify without uncertainity as well
                # we consider only the mistakes done by uncertainity implementation
                # but we have to consider OOD ground truth now
                if gt_label not in gt_matched and not gt_segms[gt_label]['category_id'] == 50:
                    continue

                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue

                if not gt_segms[gt_label]['category_id'] == 50:
                    upq_stat[0].fn += 1
                else:
                    upq_stat[1].fn += 1

            # count false positives
            pred_fp_label_id_map_ood = {}
            for pred_label_ood, pred_info in pred_segms_ood.items():
                if pred_label_ood in pred_matched_ood:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map_ood.get((VOID, pred_label_ood), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map_ood.get((crowd_labels_dict[pred_info['category_id']], pred_label_ood), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                pred_fp_label_id_map_ood[pred_label_ood] = pred_info['category_id']
            # do not consider if the id is already considered as fp without uncertainity estimation
            ood_fp_labels = set(pred_fp_label_id_map_ood.keys()) - set(pred_fp_label_id_map.keys())

            for label in ood_fp_labels:
                category_id = pred_fp_label_id_map_ood[label]
                if not category_id == 50:
                    upq_stat[0].fp += 1
                else:
                    upq_stat[1].fp += 1

    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return (pq_stat, upq_stat)


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, evaluate_ood):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories, evaluate_ood))
        processes.append(p)
    pq_stat = PQStat()
    upq_stat = UPQStat()
    for p in processes:
        pq, upq = p.get()
        pq_stat += pq
        if evaluate_ood:
            upq_stat += upq
    return pq_stat, upq_stat


def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None, evaluate_ood=False):
    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    if 'annotations_ood' in pred_json:
        pred_annotations_ood = {el['image_id']: el for el in pred_json['annotations_ood']}
    else:
        pred_annotations_ood = None
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id in pred_annotations:
            # raise Exception('no prediction for the image with id: {}'.format(image_id))
            if pred_annotations_ood is not None:
                matched_annotations_list.append((gt_ann, pred_annotations[image_id], pred_annotations_ood[image_id]))
            else:
                matched_annotations_list.append((gt_ann, pred_annotations[image_id], None))


    pq_stat, upq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, evaluate_ood)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    if evaluate_ood:
        upq_result = upq_stat.upq_average()

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )
    if evaluate_ood:
        results["OOD"] = upq_result
    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json_file', type=str,
                        help="JSON file with ground truth data")
    parser.add_argument('--pred_json_file', type=str,
                        help="JSON file with predictions data")
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground turth COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with prediction COCO format segmentations. \
                              Default: X if the corresponding json file is X.json")
    args = parser.parse_args()
    pq_compute(args.gt_json_file, args.pred_json_file, args.gt_folder, args.pred_folder)
