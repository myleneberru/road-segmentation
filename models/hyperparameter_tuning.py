"""""
Tunes the hyperparameters of the efficientnet B2 model. It uses the full dataset but different seeds than testing
"""""

from __future__ import print_function, division
from train import train_fn
from class_dataset import RoadSegDataset
from helper import *

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random as rd
from torch.optim import lr_scheduler
from torchvision import models
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold, train_test_split
from PIL import ImageFile

# Hyperparameter tuning libraries
import neptune
import neptune.integrations.optuna as optuna_utils
import optuna

ImageFile.LOAD_TRUNCATED_IMAGES = True
# To GPU is available otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def initialize_model(frozen_layers):
    base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    pretrained_weights = '/home/smartcervix/Documents/luca/Results/pretraining/EfficientNetB2_NAdam_batch_16_scheduler_' \
                         'ReduceLROnPlateau_lr_0.0_wd_0.0_beta1_0.9_beta2_0.99_factor_0.7_patience_5_step_None_gamma_' \
                         'None/efficient-epoch20-acc0.63.pth'
    base_model = change_classifier_layer(base_model, 'EfficientNetB2')
    checkpoint = torch.load(pretrained_weights)
    base_model.load_state_dict(checkpoint)
    base_model = change_classifier_layer(base_model, 'EfficientNetB2')
    base_model = freeze_layers(base_model, 'EfficientNetB2', frozen_layers)
    base_model.dropout = nn.Dropout(0.6)
    return base_model


def create_config_uniform(trial):
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW', 'RAdam', 'NAdam'])
    batchsize = trial.suggest_categorical('batchsize', [8, 16])
    scheduler = trial.suggest_categorical('scheduler', ['StepLR', 'ReduceLROnPlateau'])
    lr = trial.suggest_uniform('lr', 0.00001, 0.001)
    weight_decay = trial.suggest_uniform('weight_decay', 1e-4, 1e-2)
    frozen_layers = trial.suggest_categorical('frozen_layers', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    config = {'num_classes': 3, 'optimizer': optimizer, 'batchsize': batchsize, 'lr': lr, 'weight_decay': weight_decay,
              'scheduler': scheduler, 'frozen_layers': frozen_layers}

    if optimizer == 'SGD':
        config['momentum'] = trial.suggest_float('momentum', 0.74, 0.99, step=0.05)

    else:
        config['beta1'] = trial.suggest_float('beta1', 0.9, 0.99, step=0.01)
        config['beta2'] = trial.suggest_float('beta2', 0.9, 0.99, step=0.01)
        if optimizer == 'NAdam':
            config['momentum_decay'] = trial.suggest_uniform('momentum_decay', 4e-6, 4e-2)

    if scheduler == 'StepLR':
        config['step_size'] = trial.suggest_int('step_sizeLR', 1, 15, 1)
        config['gamma'] = trial.suggest_float('gammaLR', 0.1, 0.9, step=0.1)

    elif scheduler == 'ReduceLROnPlateau':
        config['factor'] = trial.suggest_float('factor', 0.1, 0.9, step=0.1)
        config['patience'] = trial.suggest_int('patience', 1, 15, 1)

    return config


def initialize_optim_scheduler(model, config):
    if config['optimizer'] == 'SGD':
        opt = optim.SGD(model.parameters(), momentum=config['momentum'], lr=config['lr'],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        opt = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
                               betas=(config['beta1'], config['beta2']))
    elif config['optimizer'] == 'AdamW':
        opt = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
                                betas=(config['beta1'], config['beta2']))
    elif config['optimizer'] == 'RAdam':
        opt = optim.RAdam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
                                betas=(config['beta1'], config['beta2']))
    elif config['optimizer'] == 'NAdam':
        opt = optim.NAdam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
                                betas=(config['beta1'], config['beta2']), momentum_decay=config['momentum_decay'])
    else:
        opt = None
        print('The selected optimizer is not supported.')

    if config['scheduler'] == 'StepLR':
        sch = lr_scheduler.StepLR(optimizer=opt, step_size=config['step_size'], gamma=config['gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        sch = lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=config['factor'],
                                             patience=config['patience'])
    elif config['scheduler'] == 'CyclicLR':
        sch = lr_scheduler.CyclicLR(optimizer=opt, base_lr=1e-5, max_lr=1e-1, mode='triangular2')
    elif config['scheduler'] == 'None':
        return opt
    else:
        sch = None
        print('The selected scheduler is not supported.')

    return opt, sch


def run_trial(config, datdef run_trial(config, data_dir):

    print("Initializing new trial")

    loss = []
    acc = []
    criterion_ = nn.CrossEntropyLoss()
    epochs = 30
    # images_dataset = CervicalLesionDataset(data_dir, 'EfficientNetB2')
    for seed in range(40, 43, 1):
        torch.manual_seed(seed)
        rd.seed(seed)
        np.random.seed(seed)
        # Initializing cross-validation
        k_folds = 5
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        images_dataset = CervicalLesionDataset(data_dir, 'EfficientNetB2')
        for fold, (tmp_train_ids, test_ids) in enumerate(kfold.split(X=images_dataset, y=images_dataset.class_labels)):
            val_ratio = 1 / k_folds
            val_size = int(np.ceil(val_ratio * images_dataset.__len__()))
            strat_class_labels = []
            for tmp_index in tmp_train_ids:
                strat_class_labels.append(images_dataset.__get_labels__(tmp_index))

            train_ids, val_ids = train_test_split(tmp_train_ids, test_size=val_size, shuffle=True,
                                                  stratify=strat_class_labels, random_state=seed)

            images_dataset.__set_train_ids__(train_ids)
            images_dataset.__set_val_ids__(val_ids)
            images_dataset.__set_test_ids__(test_ids)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Data loading
            train_loader = data.DataLoader(images_dataset, batch_size=config['batchsize'], sampler=train_subsampler,
                                           num_workers=4, pin_memory=True)
            val_loader = data.DataLoader(images_dataset, batch_size=config['batchsize'], sampler=val_subsampler, num_workers=4,
                                         pin_memory=True)
            test_loader = data.DataLoader(images_dataset, batch_size=1, sampler=test_subsampler, pin_memory=True)

            # Initialize network
            model = initialize_model(frozen_layers=config['frozen_layers']).to(device)

            # Optimizer and scheduler
            optimizer_, scheduler_ = initialize_optim_scheduler(model, config)
            # Training the model and reporting the best epoch according to validation
            best_model, best_epoch = train_model(False, train_loader, len(train_ids), val_loader,
                                                 len(val_ids), device, model, criterion_, optimizer_,
                                                 scheduler_, epochs, False)
            _loss, _acc = test_model(test_loader, len(test_ids), device, best_model, criterion_)
            loss.append(_loss)
            acc.append(_acc)

    return sum(loss)/len(loss), sum(acc)/len(acc)
a_dir):

    print("Initializing new trial")

    loss = []
    acc = []
    criterion_ = nn.CrossEntropyLoss()
    epochs = 30
    # images_dataset = CervicalLesionDataset(data_dir, 'EfficientNetB2')
    for seed in range(40, 43, 1):
        torch.manual_seed(seed)
        rd.seed(seed)
        np.random.seed(seed)
        # Initializing cross-validation
        k_folds = 5
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        images_dataset = CervicalLesionDataset(data_dir, 'EfficientNetB2')
        for fold, (tmp_train_ids, test_ids) in enumerate(kfold.split(X=images_dataset, y=images_dataset.class_labels)):
            val_ratio = 1 / k_folds
            val_size = int(np.ceil(val_ratio * images_dataset.__len__()))
            strat_class_labels = []
            for tmp_index in tmp_train_ids:
                strat_class_labels.append(images_dataset.__get_labels__(tmp_index))

            train_ids, val_ids = train_test_split(tmp_train_ids, test_size=val_size, shuffle=True,
                                                  stratify=strat_class_labels, random_state=seed)

            images_dataset.__set_train_ids__(train_ids)
            images_dataset.__set_val_ids__(val_ids)
            images_dataset.__set_test_ids__(test_ids)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Data loading
            train_loader = data.DataLoader(images_dataset, batch_size=config['batchsize'], sampler=train_subsampler,
                                           num_workers=4, pin_memory=True)
            val_loader = data.DataLoader(images_dataset, batch_size=config['batchsize'], sampler=val_subsampler, num_workers=4,
                                         pin_memory=True)
            test_loader = data.DataLoader(images_dataset, batch_size=1, sampler=test_subsampler, pin_memory=True)

            # Initialize network
            model = initialize_model(frozen_layers=config['frozen_layers']).to(device)

            # Optimizer and scheduler
            optimizer_, scheduler_ = initialize_optim_scheduler(model, config)
            # Training the model and reporting the best epoch according to validation
            best_model, best_epoch = train_model(False, train_loader, len(train_ids), val_loader,
                                                 len(val_ids), device, model, criterion_, optimizer_,
                                                 scheduler_, epochs, False)
            _loss, _acc = test_model(test_loader, len(test_ids), device, best_model, criterion_)
            loss.append(_loss)
            acc.append(_acc)

    return sum(loss)/len(loss), sum(acc)/len(acc)


def objective(trial):
    config = create_config_uniform(trial)
    dataset_name = 'dataset_multiclass_101_infos'
    data_dir = os.path.join('/home/smartcervix/Documents/mehdi/Data/old_data', dataset_name)
    loss, acc = run_trial(config, data_dir)
    return loss, acc


def generate_param_report(logfile, param_):
    log_file = open(logfile, 'w')
    for key, val in param_.items():
        log_file.write(key+':'+str(val)+'\n')
    log_file.close()


def test_model(test_dataset, test_size, device, model, criterion):
    # Softmax used in multiclass case
    softmax = nn.Softmax(dim=1)
    model = model.to(device)
    running_loss = 0.0
    running_corrects = 0

    model.eval()
    with torch.no_grad():
        for ii, sample in enumerate(test_dataset):
            label = sample['labels']
            input_ = sample['images'].to(device)
            input_label = label.to(device)

            outputs = model(input_)
            loss = criterion(outputs, input_label)
            probabilities = softmax(outputs)
            _, preds = torch.max(probabilities, 1)
            _probas = probabilities.cpu().data.numpy()[0]

            # Have whole batch statistics
            running_loss += loss.item() * input_.size(0)
            running_corrects += torch.sum(preds == input_label.data)
    # Print statistics
    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects / test_size

    return epoch_loss, epoch_acc



if __name__ == '__main__':
    run = neptune.init_run(
        project='smartcervix',  # TODO: insert your neptune project
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNTY0M2ZjNy05YjFmLTRlYTctYjg1ZS05ZmY2OTdjZjc1ZWYifQ==',  # TODO: insert your neptune token
    )
    neptune_callback = optuna_utils.NeptuneCallback(run, log_plot_slice=False, log_plot_contour=False, log_plot_edf=False, log_plot_parallel_coordinate=False)
    study = optuna.create_study(directions=['minimize', 'maximize'], study_name="hyperparameter_tuning", load_if_exists=False,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100, callbacks=[neptune_callback])
    run.stop()
