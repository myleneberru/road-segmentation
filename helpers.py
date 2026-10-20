import numpy as np
import os
import pickle
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
import torch.utils.data as data
import time
from models.unet.unet_class import Unet

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 100
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

def load_model(model_path, device = "cpu"):

    model = Unet().to(device)
    model.load_state_dict(torch.load(model_path,map_location = torch.device(device)))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.33298134, 0.33009373, 0.29579783], [0.19390678, 0.18714967, 0.18701911])
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image

def apply_threshold(prediction, threshold=0.5):
    prediction = prediction > threshold
    return prediction

def print_advancement(i):
    progress_e = 100 * i / 50
    print(f' Progress : {progress_e : .2f}%', end='\r')


def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou_score = np.sum(intersection) / np.sum(union )
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


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print("Loading " + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print("File " + image_filename + " does not exist")

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return np.asarray(data)

def predict(data, model):


    for image in data:

        print_advancement(i)

        image = preprocess_image(image)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            prediction = model(image)

        # Convert prediction to an image format
        # Update this part based on your prediction format
        prediction_image = prediction.squeeze().cpu().numpy()
        binary_prediction = apply_threshold(prediction_image)

        # Convert binary prediction to image and save
        binary_prediction_image = Image.fromarray((binary_prediction * 255).astype(np.uint8))
        save_path = os.path.join(save_dir, f'prediction_{i}.png')
        binary_prediction_image.save(save_path)