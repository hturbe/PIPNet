import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments

"""


class Args:
    def __init__(
        self,
        dataset="CUB-200-2011",
        validation_size=0.0,
        net="convnext_tiny_26",
        batch_size=64,
        batch_size_pretrain=128,
        epochs=60,
        epochs_pretrain=10,
        optimizer="Adam",
        lr=0.05,
        lr_block=0.0005,
        lr_net=0.0005,
        weight_decay=0.0,
        disable_cuda=False,
        log_dir="./runs/run_pipnet",
        num_features=0,
        image_size=224,
        state_dict_dir_net="",
        freeze_epochs=10,
        dir_for_saving_images="visualization_results",
        disable_pretrained=False,
        weighted_loss=False,
        seed=1,
        gpu_ids="",
        num_workers=8,
        bias=False,
        extra_test_image_folder="./experiments",
    ):
        self.dataset = dataset
        self.validation_size = validation_size
        self.net = net
        self.batch_size = batch_size
        self.batch_size_pretrain = batch_size_pretrain
        self.epochs = epochs
        self.epochs_pretrain = epochs_pretrain
        self.optimizer = optimizer
        self.lr = lr
        self.lr_block = lr_block
        self.lr_net = lr_net
        self.weight_decay = weight_decay
        self.disable_cuda = disable_cuda
        self.log_dir = log_dir
        self.num_features = num_features
        self.image_size = image_size
        self.state_dict_dir_net = state_dict_dir_net
        self.freeze_epochs = freeze_epochs
        self.dir_for_saving_images = dir_for_saving_images
        self.disable_pretrained = disable_pretrained
        self.weighted_loss = weighted_loss
        self.seed = seed
        self.gpu_ids = gpu_ids
        self.num_workers = num_workers
        self.bias = bias
        self.extra_test_image_folder = extra_test_image_folder


def get_args() -> Args:
    parser = argparse.ArgumentParser("Train a PIP-Net")
    parser.add_argument("--dataset", type=str, default="FunnyBirds", help="Data set on PIP-Net should be trained")
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.0,
        help="Split between training and validation set. Can be zero when there is a separate test or validation directory. Should be between 0 and 1. Used for partimagenet (e.g. 0.2)",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="convnext_tiny_26",
        help="Base network used as backbone of PIP-Net. Default is convnext_tiny_26 with adapted strides to output 26x26 latent representations. Other option is convnext_tiny_13 that outputs 13x13 (smaller and faster to train, less fine-grained). Pretrained network on iNaturalist is only available for resnet50_inat. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, convnext_tiny_26 and convnext_tiny_13.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs",
    )
    parser.add_argument(
        "--batch_size_pretrain",
        type=int,
        default=16,
        help="Batch size when pretraining the prototypes (first training stage)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="The number of epochs PIP-Net should be trained (second training stage)"
    )
    parser.add_argument(
        "--epochs_pretrain",
        type=int,
        default=5,
        help="Number of epochs to pre-train the prototypes (first training stage). Recommended to train at least until the align loss < 1",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="The optimizer that should be used when training PIP-Net"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="The optimizer learning rate for training the weights from prototypes to classes",
    )
    parser.add_argument(
        "--lr_block",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for training the last conv layers of the backbone",
    )
    parser.add_argument(
        "--lr_net",
        type=float,
        default=0.0005,
        help="The optimizer learning rate for the backbone. Usually similar as lr_block.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay used in the optimizer")
    parser.add_argument("--disable_cuda", action="store_true", help="Flag that disables GPU usage if set")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs/run_FunnyBirdsv2",
        help="The directory in which train progress should be logged",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=0,
        help="Number of prototypes. When zero (default) the number of prototypes is the number of output channels of backbone. If this value is set, then a 1x1 conv layer will be added. Recommended to keep 0, but can be increased when number of classes > num output channels in backbone.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input images will be resized to --image_size x --image_size (square). Code only tested with 224x224, so no guarantees that it works for different sizes.",
    )
    parser.add_argument(
        "--state_dict_dir_net",
        type=str,
        default="",
        help="The directory containing a state dict with a pretrained PIP-Net. E.g., ./runs/run_pipnet/checkpoints/net_trained",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=5,
        help="Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)",
    )
    parser.add_argument(
        "--dir_for_saving_images",
        type=str,
        default="visualization_results",
        help="Directoy for saving the prototypes and explanations",
    )
    parser.add_argument(
        "--disable_pretrained",
        action="store_true",
        help="When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).",
    )
    parser.add_argument(
        "--weighted_loss",
        action="store_true",
        help="Flag that weights the loss based on the class balance of the dataset. Recommended to use when data is imbalanced. ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html",
    )
    parser.add_argument("--gpu_ids", type=str, default="", help="ID of gpu. Can be separated with comma")
    parser.add_argument("--num_workers", type=int, default=8, help="Num workers in dataloaders.")
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Flag that indicates whether to include a trainable bias in the linear classification layer.",
    )
    parser.add_argument(
        "--extra_test_image_folder",
        type=str,
        default="./experiments",
        help="Folder with images that PIP-Net will predict and explain, that are not in the training or test set. E.g. images with 2 objects or OOD image. Images should be in subfolder. E.g. images in ./experiments/images/, and argument --./experiments",
    )

    args = parser.parse_args()
    if len(args.log_dir.split("/")) > 2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    # return Args(
    #     dataset=args.dataset,
    #     validation_size=args.validation_size,
    #     net=args.net,
    #     batch_size=args.batch_size,
    #     batch_size_pretrain=args.batch_size_pretrain,
    #     epochs=args.epochs,
    #     epochs_pretrain=args.epochs_pretrain,
    #     optimizer=args.optimizer,
    #     lr=args.lr,
    #     lr_block=args.lr_block,
    #     lr_net=args.lr_net,
    #     weight_decay=args.weight_decay,
    #     disable_cuda=args.disable_cuda,
    #     log_dir=args.log_dir,
    #     num_features=args.num_features,
    #     image_size=args.image_size,
    #     state_dict_dir_net=args.state_dict_dir_net,
    #     freeze_epochs=args.freeze_epochs,
    #     dir_for_saving_images=args.dir_for_saving_images,
    #     disable_pretrained=args.disable_pretrained,
    #     weighted_loss=args.weighted_loss,
    #     seed=args.seed,
    #     gpu_ids=args.gpu_ids,
    #     num_workers=args.num_workers,
    #     bias=args.bias,
    #     extra_test_image_folder=args.extra_test_image_folder,
    # )

    args = parser.parse_args()
    if len(args.log_dir.split("/")) > 2:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + "/args.txt", "w") as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write("{}: {}\n".format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + "/args.pickle", "wb") as f:
        pickle.dump(args, f)


def get_optimizer_nn(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if "resnet50" in args.net:
        # freeze resnet50 except last convolutional layer
        for name, param in net.module._net.named_parameters():
            if "layer4.2" in name:
                params_to_train.append(param)
            elif "layer4" in name or "layer3" in name:
                params_to_freeze.append(param)
            elif "layer2" in name:
                params_backbone.append(param)
            else:  # such that model training fits on one gpu.
                param.requires_grad = False
                # params_backbone.append(param)

    elif "convnext" in args.net:
        print("chosen network is convnext", flush=True)
        for name, param in net.module._net.named_parameters():
            if "features.7.2" in name:
                params_to_train.append(param)
            elif "features.7" in name or "features.6" in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    elif "dino" in args.net:
        print("chosen network is DINO", flush=True)
        for name, param in net.module._net.named_parameters():
            if "blocks.23"in name:
                params_to_train.append(param)
            # elif "blocks.23" in name:
            #     print("hello block 23",name)
            #     params_to_freeze.append(param)
            else:
                params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)
    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if "weight" in name:
            classification_weight.append(param)
        elif "multiplier" in name:
            param.requires_grad = False
        else:
            if args.bias:
                classification_bias.append(param)

    paramlist_net = [
        {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
        {"params": params_to_freeze, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
        {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
        {"params": net.module._add_on.parameters(), "lr": args.lr_block * 10.0, "weight_decay_rate": args.weight_decay},
    ]

    paramlist_classifier = [
        {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
        {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]

    if args.optimizer == "Adam":
        optimizer_net = torch.optim.AdamW(paramlist_net, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")
