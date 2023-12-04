import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class RoadSegDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        self.train_ids = []
        self.test_ids = []
        self.val_ids = []

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_transform = self.get_img_transform(index)
        mask_transform = self.get_mask_transform(index)

        image = img_transform(image)
        mask = mask_transform(mask)

        mask = (mask > 0).to(torch.float32)

        return image, mask

    def __len__(self):
        return len(self.images)

    def __set_train_ids__(self, train_ids):
        self.train_ids = train_ids

    def __set_val_ids__(self, val_ids):
        self.val_ids = val_ids

    def __set_test_ids__(self, test_ids):
        self.test_ids = test_ids

    def __get_train_ids__(self):
        return self.train_ids

    def get_img_transform(self, index):
        """
        Returns the transformation to apply to the image at the given index
        :param index:
        :return:
        """

        if index in self.train_ids:
            return transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.RandomRotation(degrees=(0, 90)),
                transforms.ToTensor(),
                #transforms.Normalize([0.33298134, 0.33009373, 0.29579783],[0.19390678, 0.18714967, 0.18701911])
            ])
        else:  # For validation and test images, no augmentation is applied
            return transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.33298134, 0.33009373, 0.29579783],[0.19390678, 0.18714967, 0.18701911])
            ])

    def get_mask_transform(self, index):
        """
        Returns the transformation to apply to the mask at the given index
        :param index:
        :return:
        """

        if index in self.train_ids:
            return transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.RandomRotation(degrees=(0, 90)),
                transforms.ToTensor(),
            ])
        else:  # For validation and test images, no augmentation is applied
            return transforms.Compose([
                transforms.ToTensor(),
            ])
