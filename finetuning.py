from __future__ import print_function, division
from models.train import train_fn
from models.class_dataset import RoadSegDataset
from helper import *
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



def test_model(loader, test_size, device, model_tt, loss_fn, threshold=0.5):
    running_loss = 0.0

    model_tt.eval()

    ground_truth = []
    predictions = []
    inference_time = []

    for inputs, masks in loader:
        inputs = inputs.to(device)
        inputs = inputs.float()
        masks = masks.to(device)
        masks = masks.squeeze()
        start = time.time()

        with torch.set_grad_enabled(False):

            outputs = model_tt(inputs).squeeze()
            loss = loss_fn(outputs, masks)
            preds = (outputs > threshold).float()

        ground_truth.append(masks.cpu().numpy())
        predictions.append(preds.cpu().numpy())
        inference_time.append(time.time() - start)

        running_loss += loss.item() * inputs.size(0)
        inference_time.append(time.time() - start)

    epoch_loss = running_loss / test_size
    print("Test loss={:.3f}".format(epoch_loss))

    return ground_truth, predictions, inference_time


# HyperParameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
lr_opt = 1e-4
batch_size = 16
num_epochs = 10
num_workers = 2

load_model = False
image_height = 400
image_width = 400
opti_opt = 'Adam'
wd_opt = 0.00957460777341424
schedu_opt = 'Step'
step_size_opt = 10
gamma_opt = 0.2
save_model = True
model_name = 'unet'
train_img_dir = "data/training/train_images"
train_masks_dir = "data/training/train_masks"
save_path = os.path.join('models', model_name, 'Results')


# ecrire une fonction qui load le bon model en fonction de model  name
loss_fn = nn.BCEWithLogitsLoss()

if __name__ == '__main__':

    dataset = RoadSegDataset(train_img_dir, train_masks_dir)
    print("Running experiments with: ", model_name)

    report = {'iou': [], 'dice': [], 'f1': [], 'pixel accuracy': [], 'inference_time': []}

    n_splits = 2
    kf = KFold(n_splits=n_splits)

    for fold, (train_ids, test_ids) in enumerate(kf.split(np.arange(dataset.__len__()))):
        # Split test_index into validation and test sets if necessary

        ground_truth = []
        predictions = []
        probabilities = []
        inf_time = []

        split = int(len(test_ids) * 0.5)
        val_ids, test_ids = test_ids[:split], test_ids[split:]

        dataset.__set_train_ids__(train_ids)
        dataset.__set_val_ids__(val_ids)
        dataset.__set_test_ids__(test_ids)

        print(len(train_ids), len(test_ids), len(val_ids))

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Create data loaders
        train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        test_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        save_flag = True

        # Initialize network
        model = Unet().to(device)

        # Model Optimizer
        if opti_opt == 'Adam':
            optimizer_ft = optim.Adam(model.parameters(), lr=lr_opt, weight_decay=wd_opt)
        elif opti_opt == 'RAdam':
            optimizer_ft = optim.RAdam(model.parameters(), lr=lr_opt, weight_decay=wd_opt,
                                       betas=(0.94, 0.99))

        # Model Scheduler
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size_opt, gamma=gamma_opt)

        # Training the model and reporting the best epoch according to validation

        print(f'Training Started for fold number {fold} ')

        best_model, best_epoch = train_fn(train_loader, len(train_ids), val_loader, len(val_ids), device, model,
                                          loss_fn, optimizer_ft,
                                          exp_lr_scheduler, num_epochs, save_path, save_model, fold)

        print(f'Training Ended for fold number {fold} ')

        print(f'Testing Started for fold number {fold}')

        _ground_truth, _predictions, _inf_time = test_model(test_loader, len(test_ids), device, best_model, loss_fn,
                                                            0.5)
        print(f'Testing Ended for fold number {fold}')

        ground_truth += _ground_truth
        predictions += _predictions
        inf_time += _inf_time

        # Assuming predictions and ground_truth are lists of lists
        flattened_predictions = [item for sublist in predictions for item in sublist]
        flattened_ground_truth = [item for sublist in ground_truth for item in sublist]

        # Now, flattened_predictions and flattened_ground_truth are single lists containing all the predictions and ground truths respectively

        iou, dice, pixel_acc, f = segmentation_metrics(flattened_ground_truth, flattened_predictions)

        report['iou'].append(iou)
        report['dice'].append(dice)
        report['pixel accuracy'].append(pixel_acc)
        report['f1'].append(f)
        report['inference_time'] += inf_time
        print("-------------------------------------")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    report_segmentation_performance(report, save_path)

    print("Tbarkellah 3la si rachid")
