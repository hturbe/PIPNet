import argparse
import os
import random
from typing import Dict, Tuple

import numpy as np
import pyrootutils
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import Tensor

pyrootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)
from util.FunnybirdsDataset import FunnyBirdsDataset


def get_data(args: argparse.Namespace):
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset == "CUB-200-2011":
        return get_birds(
            True,
            "./data/CUB_200_2011/dataset/train_crop",
            "./data/CUB_200_2011/dataset/train",
            "./data/CUB_200_2011/dataset/test_crop",
            args.image_size,
            args.seed,
            args.validation_size,
            "./data/CUB_200_2011/dataset/train",
            "./data/CUB_200_2011/dataset/test_full",
        )
    if args.dataset == "FunnyBirds":
        return get_FunnyBirds(
            True,
            "./data/FunnyBirds/",
            "./data/FunnyBirds/",
            "./data/FunnyBirds/",
            args.image_size,
            args.seed,
            args.validation_size,
        )
    if args.dataset == "pets":
        return get_pets(
            True,
            "./data/PETS/dataset/train",
            "./data/PETS/dataset/train",
            "./data/PETS/dataset/test",
            args.image_size,
            args.seed,
            args.validation_size,
        )
    if args.dataset == "partimagenet":  # use --validation_size of 0.2
        return get_partimagenet(
            True,
            "./data/partimagenet/dataset/all",
            "./data/partimagenet/dataset/all",
            None,
            args.image_size,
            args.seed,
            args.validation_size,
        )
    if args.dataset == "CARS":
        return get_cars(
            True,
            "./data/cars/dataset/train",
            "./data/cars/dataset/train",
            "./data/cars/dataset/test",
            args.image_size,
            args.seed,
            args.validation_size,
        )
    if args.dataset == "grayscale_example":
        return get_grayscale(
            True,
            "./data/train",
            "./data/train",
            "./data/test",
            args.image_size,
            args.seed,
            args.validation_size,
        )
    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')


def get_dataloaders(args: argparse.Namespace, device):
    """
    Get data loaders
    """
    # Obtain the dataset
    (
        trainset,
        trainset_pretraining,
        trainset_normal,
        trainset_normal_augment,
        projectset,
        testset,
        testset_projection,
        classes,
    ) = get_data(args)

    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None

    num_workers = args.num_workers

    if args.weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor(
            [(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)]
        )
        weight = 1.0 / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        to_shuffle = False

    pretrain_batchsize = args.batch_size_pretrain

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(args.seed),
        drop_last=True,
    )
    if trainset_pretraining is not None:
        trainloader_pretraining = torch.utils.data.DataLoader(
            trainset_pretraining,
            batch_size=pretrain_batchsize,
            shuffle=to_shuffle,
            sampler=sampler,
            pin_memory=cuda,
            num_workers=num_workers,
            worker_init_fn=np.random.seed(args.seed),
            drop_last=True,
        )

    else:
        trainloader_pretraining = torch.utils.data.DataLoader(
            trainset,
            batch_size=pretrain_batchsize,
            shuffle=to_shuffle,
            sampler=sampler,
            pin_memory=cuda,
            num_workers=num_workers,
            worker_init_fn=np.random.seed(args.seed),
            drop_last=True,
        )

    trainloader_normal = torch.utils.data.DataLoader(
        trainset_normal,
        batch_size=args.batch_size,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(args.seed),
        drop_last=True,
    )
    trainloader_normal_augment = torch.utils.data.DataLoader(
        trainset_normal_augment,
        batch_size=args.batch_size,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(args.seed),
        drop_last=True,
    )

    projectloader = torch.utils.data.DataLoader(
        projectset,
        batch_size=1,
        shuffle=False,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(args.seed),
        drop_last=False,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(args.seed),
        drop_last=False,
    )
    test_projectloader = torch.utils.data.DataLoader(
        testset_projection,
        batch_size=1,
        shuffle=False,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(args.seed),
        drop_last=False,
    )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return (
        trainloader,
        trainloader_pretraining,
        trainloader_normal,
        trainloader_normal_augment,
        projectloader,
        testloader,
        test_projectloader,
        classes,
    )


def create_datasets(
    transform1,
    transform2,
    transform_no_augment,
    num_channels: int,
    train_dir: str,
    project_dir: str,
    test_dir: str,
    seed: int,
    validation_size: float,
    train_dir_pretrain=None,
    test_dir_projection=None,
    transform1p=None,
):
    trainvalset = FunnyBirdsDataset(train_dir, split="train")
    classes = [trainvalset.classes[i]["class_idx"] for i in range(len(trainvalset.classes))]
    targets = np.arange(50)
    indices = list(range(len(trainvalset)))

    train_indices = indices

    testset = FunnyBirdsDataset(test_dir, transform=transform_no_augment, split="test")

    trainset = TwoAugSupervisedDataset(trainvalset, transform1=transform1, transform2=transform2)

    trainset_normal = FunnyBirdsDataset(train_dir, transform=transform_no_augment, split="train")

    trainset_normal_augment = FunnyBirdsDataset(
        train_dir, transform=transforms.Compose([transform1, transform2]), split="train"
    )
    projectset = FunnyBirdsDataset(project_dir, transform=transform_no_augment, split="train")

    if test_dir_projection is not None:
        testset_projection = FunnyBirdsDataset(test_dir_projection, transform=transform_no_augment, split="test")
    else:
        testset_projection = testset
    if train_dir_pretrain is not None:
        trainvalset_pr = FunnyBirdsDataset(train_dir_pretrain, split="train")
        trainset_pretraining = TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1p, transform2=transform2)
    else:
        trainset_pretraining = None

    return (
        trainset,
        trainset_pretraining,
        trainset_normal,
        trainset_normal_augment,
        projectset,
        testset,
        testset_projection,
        classes,
    )


def get_FunnyBirds(
    augment: bool,
    train_dir: str,
    project_dir: str,
    test_dir: str,
    img_size: int,
    seed: int,
    validation_size: float,
):
    transform_no_augment = transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor()])

    if augment:
        transform1 = transforms.Compose(
            [
                transforms.Resize(size=(img_size + 48, img_size + 48)),
                TrivialAugmentWideNoColor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(img_size + 8, scale=(0.95, 1.0)),
            ]
        )

        transform2 = transforms.Compose(
            [
                TrivialAugmentWideNoShape(),
                transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
                transforms.ToTensor(),
            ]
        )
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(
        transform1,
        transform2,
        transform_no_augment,
        3,
        train_dir,
        project_dir,
        test_dir,
        seed,
        validation_size,
    )


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""

    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes

        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)


# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True),
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (
                8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(),
                False,
            ),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
