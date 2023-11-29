import torch
import copy


def train_fn(train_loader, train_size, val_loader, val_size, device, model, criterion, optimizer, scheduler, num_epochs,
             save_path, save_model, fold):
    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = float('inf')
    best_epoch = 0

    print("Beginning of training")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, masks in (train_loader if phase == 'train' else val_loader):

                inputs = inputs.to(device)
                masks = masks.to(device)
                masks = masks.squeeze()
                inputs = inputs.float()


                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    print (f'pred shape : {outputs.shape}')
                    print(f'mask shape : {masks.shape}')
                    loss = criterion(outputs, masks)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / (train_size if phase == 'train' else val_size)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # Deep copy the model
            if phase == 'val' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best epoch found with validation loss: ', best_epoch)
    model.load_state_dict(best_model_wts)
    if save_model:
        torch.save(model.state_dict(), f'{save_path}/model_epoch{best_epoch}_fold_{fold}.pth')
    # Load best model weights

    return model, best_epoch
