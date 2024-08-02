from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd
import scipy.io as sio
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (
    verify_str_arg,
)
from torchvision.datasets.vision import VisionDataset

DATASET_CLASSES = {
    "n": 0,  # benign tissues.
    "aca": 1,  # adenocarcinomas
    "scc": 2,  # squamous cell carcinomas
}


# https://github.com/tampapath/lung_colon_image_set
class LungColonDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader=default_loader,
        photometric_transforms=None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self.root = Path(root)
        self.loader = loader

        if not self._check_exists():
            raise RuntimeError("Dataset not found")

        self._load_metadata()

    def _load_metadata(self):
        self._samples = pd.read_csv(self.root / "split.csv")
        if self._split == "train":
            self.data = self._samples[self._samples["split"] == "train"]
        elif self._split == "test":
            self.data = self._samples[self._samples["split"] == "test"]
        print("Data shape", self.data.shape)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        sample = self.data.iloc[idx]
        path = Path(self.root) / sample.filepath
        target = DATASET_CLASSES[sample.target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self) -> bool:
        if not (self.root).is_dir():
            return False
        else:
            return True
