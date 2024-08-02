import os
import ssl
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

dict_disease = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
    "UNK": "None of the others",
}

malign_categories = {
    "MEL": "Melanoma",
    "SCC": "Squamous cell carcinoma",
    "BCC": "Basal cell carcinoma",
}


class IsicDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader=default_loader,
        download=False,
        photometric_transforms=None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.loader = loader
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.photometric_transforms = photometric_transforms

        # if not self._check_integrity():
        #     raise RuntimeError(
        #         "Dataset not found or corrupted. You can use download=True to download it"
        #     )

        self.data: Any = []
        self.targets = []

        self._load_metadata()

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def _load_metadata(self):
        # images = pd.read_csv(
        #     os.path.join(self.root, "CUB_200_2011", "images.txt"),
        #     sep=" ",
        #     names=["img_id", "filepath"],
        # )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "ISIC_2019_Training_GroundTruth.csv"),
            index_col=0,
            # names=["img_id", "target"],
        )

        # find the idx of the col with the max value in each row
        image_class_labels.loc[:, "target"] = np.argmax(image_class_labels.to_numpy(), axis=1)

        # image_class_labels.loc[:, "target"] = (
        #     image_class_labels.idxmax(axis=1).isin(malign_categories.keys()).astype(int)
        # )
        image_class_labels = image_class_labels.loc[:, "target"]
        image_class_labels.index.name = "img_id"
        image_class_labels = image_class_labels.reset_index()
        train_test_split = pd.read_csv(
            os.path.join(self.root, "data_split.csv"),
            index_col=0,
            names=["img_id", "split"],
        )

        data = train_test_split.merge(image_class_labels, on="img_id")
        data.loc[:, "filepath"] = data.apply(lambda x: f"images/{x.img_id}.jpg", axis=1)
        self.data = data

        if self.train:
            self.data = self.data[self.data.split == "train"]
        else:
            self.data = self.data[self.data.split == "test"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, sample.filepath)
        target = sample.target  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.photometric_transforms is not None:
            img_aug = self.photometric_transforms(img)
            return img, target, img_aug

        else:
            return img, target
