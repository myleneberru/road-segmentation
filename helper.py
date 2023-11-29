import numpy as np
import os
import pickle


def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_dice(pred, gt):
    intersection = np.logical_and(pred, gt)
    dice_score = 2 * np.sum(intersection) / (np.sum(pred) + np.sum(gt))
    return dice_score

def calculate_pixel_accuracy(pred, gt):

    if isinstance(pred, list):
        pred = np.array(pred)
    if isinstance(gt, list):
        gt = np.array(gt)

    pred_flat = pred.reshape(-1)  # This will create a flat array
    gt_flat = gt.reshape(-1)      # This will create a flat array

    correct = np.sum(pred_flat == gt_flat)
    total = len(pred_flat)
    return correct / total

def calculate_f1(pred, gt):
    precision = np.sum(np.logical_and(pred, gt)) / np.sum(pred)
    recall = np.sum(np.logical_and(pred, gt)) / np.sum(gt)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score



def report_segmentation_performance(reports, save_dir):
    """
    Compute the average performance of segmentation metrics across folds and write to file.
    :param reports: Dictionary containing lists of performance metrics for each fold
    :param save_dir: Directory where the performance report will be saved
    :return: None
    """

    report = "Performance Summary Across Folds:\n"
    for metric in ['iou', 'dice', 'pixel accuracy', 'f1']:
        mean_val = np.mean(reports[metric])
        std_val = np.std(reports[metric])
        report += f"{metric.upper()}: Mean = {mean_val:.4f}, Std = {std_val:.4f}\n"

    report += "Average inference time: Mean = {:.4f}, Std = {:.4f}\n".format(
        np.mean(reports['inference_time']), 
        np.std(reports['inference_time'])
    )

    print(report)

    # Save the report to a text file
    with open(os.path.join(save_dir, 'Performance_report.txt'), 'wt') as f:
        f.write(report)

    # Save the report dictionary to a file for later use
    with open(os.path.join(save_dir, 'Performance_dict.pkl'), 'wb') as f:
        pickle.dump(reports, f)

def segmentation_metrics (_ground_truth, _predictions):

    iou = calculate_iou(_predictions,_ground_truth)
    dice = calculate_dice(_predictions, _ground_truth)
    pixel_acc = calculate_pixel_accuracy(_predictions,_ground_truth)
    f = calculate_f1(_predictions, _ground_truth)

    return iou, dice, pixel_acc, f