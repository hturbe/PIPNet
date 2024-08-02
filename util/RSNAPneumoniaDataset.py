import os
import ssl
from glob import glob
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset


#  Adapted from: https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
class RSNAPneumoniaDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader=default_loader,
        photometric_transforms=False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.loader = loader
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.photometric_transforms = None

        # if train:
        #     self.transform = transforms.Compose(
        #         [
        #             transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        #             # transforms.ToTensor(),
        #         ]
        #     )
        # else:
        #     self.transform = transforms.Compose(
        #         [
        #             transforms.Resize((224, 224)),
        #             # transforms.ToTensor(),
        #         ]
        #     )

        self.data: Any = []

        self._load_metadata()

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def _load_metadata(self):
        if (not os.path.exists(os.path.join(self.root, "stage_2_train_images_png"))) | (
            not os.path.exists(os.path.join(self.root, "stage_2_test_images_png"))
        ):
            self.convert_to_png()

        list_images = glob(os.path.join(self.root, "stage_2_train_images_png/*.png"))
        image_metadata = pd.read_csv(
            os.path.join(self.root, "stage_2_train_labels.csv"),
        )
        image_metadata["filepath"] = image_metadata["patientId"].apply(
            lambda x: os.path.join("stage_2_train_images_png", f"{x}.png")
        )

        image_metadata = image_metadata.loc[:, ["patientId", "filepath", "Target"]]
        # drop duplicates
        image_metadata = image_metadata.drop_duplicates(subset=["patientId"], keep="first")
        df_split = pd.read_csv(os.path.join(self.root, "split.csv"))
        # image_metadata["is_training_img"] = 1
        self.data = image_metadata.merge(df_split, on="patientId", how="inner")

        if self.train:
            self.data = self.data[self.data.split == "train"]
        else:
            self.data = self.data[self.data.split == "test"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, sample.filepath)
        target = int(sample.Target)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        # lower_bound, upper_bound = (
        #     torch.quantile(img[img > 0], 0.005),
        #     torch.quantile(img[img > 0], 0.95),
        # )
        # image_data_pre = torch.clip(img, lower_bound, upper_bound)
        # image_data_pre = (image_data_pre - torch.min(image_data_pre)) / (
        #     torch.max(image_data_pre) - torch.min(image_data_pre)
        # )
        # image_data_pre[img == 0] = 0

        if self.target_transform is not None:
            target = self.target_transform.transform(np.array(target).reshape(1, -1)).reshape(-1)
        return img, target

    def convert_to_png(self):
        if not os.path.exists(os.path.join(self.root, "stage_2_train_images_png")):
            print("Converting train images to png...")
            os.makedirs(os.path.join(self.root, "stage_2_train_images_png"))
            list_images = glob(os.path.join(self.root, "stage_2_train_images/*.dcm"))
            save_path = os.path.join(self.root, "stage_2_train_images_png")
            # run convert dicom png with joblib in parallel
            joblib.Parallel(n_jobs=-1)(joblib.delayed(convert_dicom_png)(img, save_path) for img in list_images)
        if not os.path.exists(os.path.join(self.root, "stage_2_test_images_png")):
            print("Converting test images to png...")
            os.makedirs(os.path.join(self.root, "stage_2_test_images_png"))
            list_images = glob(os.path.join(self.root, "stage_2_test_images/*.dcm"))
            save_path = os.path.join(self.root, "stage_2_test_images_png")
            # run convert dicom png with joblib in parallel
            joblib.Parallel(n_jobs=-1)(joblib.delayed(convert_dicom_png)(img, save_path) for img in list_images)


def convert_dicom_png(dcm_path, save_path):
    patient_id = os.path.split(dcm_path)[-1].split(".")[0]
    ds = pydicom.dcmread(dcm_path)
    # Convert the pixel data to a numpy array
    image_array = ds.pixel_array

    # Create an Image object from the numpy array
    image = Image.fromarray(image_array)

    # Save the image as PNG
    image.save(os.path.join(save_path, f"{patient_id}.png"))
