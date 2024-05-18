import argparse
import os
import random
from os.path import join as pj

import pyrootutils
import torch
import torch.nn as nn

pyrootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.funny_birds.evaluation_protocols import (
    accuracy_protocol,
    background_independence_protocol,
    controlled_synthetic_data_check_protocol,
    deletion_check_protocol,
    distractibility_protocol,
    preservation_check_protocol,
    single_deletion_protocol,
    target_sensitivity_protocol,
)
from evaluation.funny_birds.explainers.explainer_wrapper import ProtoExplainer
from evaluation.funny_birds.plot_results import plot_results_funny_birds
from FunnybirdsDataset import FunnyBirdsDataset
from pipnet.pipnet import PIPNet, get_network
from util.args import get_args
from util.data import get_dataloaders

FILEPATH = os.path.dirname(os.path.abspath(__file__))


class ModelExplainerWrapper(nn.Module):
    def __init__(self, model, explainer=None):
        """
        A generic wrapper that takes any model and any explainer to putput model predictions
        and explanations that highlight important input image part.
        Args:
            model: PyTorch neural network model
            explainer: PyTorch model explainer
        """
        super().__init__()
        self.model = model
        self.explainer = explainer

    def forward(self, input):
        output = self.model(input)
        output = output[2]

        return output

    def predict(self, input):
        output = self.model.forward(input)
        breakpoint()

    def explain(self, input):
        return self.explainer.explain(self.model, input)


def main(path_sim):
    data_path = "/workspaces/PIPNet/data/FunnyBirds"
    # device = "cuda:" + str(args.gpu)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    random.seed(42)
    torch.manual_seed(42)

    # create model

    (
        trainloader,
        trainloader_pretraining,
        trainloader_normal,
        trainloader_normal_augment,
        projectloader,
        testloader,
        test_projectloader,
        classes,
    ) = get_dataloaders(args, device)


    test_dataset = FunnyBirdsDataset(
        data_path, "test", eval_funny_birds=True
    )
    test_loader2 = DataLoader(test_dataset, batch_size=64, shuffle=False)


    for sample in testloader:
        sample1 = sample
        break

    for sample in test_loader2:
        sample2 = sample
        break

    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = (
        get_network(len(classes), args)
    )

    # Create a PIP-Net
    model = PIPNet(
        num_classes=len(classes),
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    model = model.to(device=device)
    checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    # rename all key by replacing "module." with "_net"
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, strict=True)

    batch_size = 64

    model = ModelExplainerWrapper(model)
    model.eval()
    explainer = ProtoExplainer(model)

    breakpoint()
    accuracy, csdc, pc, dc, distractibility, sd, ts = -1, -1, -1, -1, -1, -1, -1

    print("Computing accuracy...")
    accuracy = accuracy_protocol(model, data_path, batch_size)
    accuracy = round(accuracy, 5)
    breakpoint()

    print("Computing controlled synthetic data check...")
    csdc = controlled_synthetic_data_check_protocol(model, explainer, data_path)

    print("Computing target sensitivity...")
    ts = target_sensitivity_protocol(model, explainer, data_path)
    ts = round(ts, 5)

    print("Computing single deletion...")
    sd = single_deletion_protocol(model, explainer, data_path)
    sd = round(sd, 5)

    print("Computing preservation check...")
    pc = preservation_check_protocol(model, explainer, data_path)

    print("Computing deletion check...")
    dc = deletion_check_protocol(model, explainer, data_path)

    print("Computing distractibility...")
    distractibility = distractibility_protocol(model, explainer, data_path)

    print("Computing background independence...")
    background_independence = background_independence_protocol(model, data_path)
    background_independence = round(background_independence, 5)

    # select completeness and distractability thresholds such that they maximize the sum of both
    max_score = 0
    best_threshold = -1
    for threshold in csdc.keys():
        max_score_tmp = (
            csdc[threshold] / 3.0
            + pc[threshold] / 3.0
            + dc[threshold] / 3.0
            + distractibility[threshold]
        )
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    print("FINAL RESULTS:")
    print("Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS")
    print(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            accuracy,
            round(csdc[best_threshold], 5),
            round(pc[best_threshold], 5),
            round(dc[best_threshold], 5),
            round(distractibility[best_threshold], 5),
            background_independence,
            sd,
            ts,
        )
    )
    print("Best threshold:", best_threshold)
    results = [
        accuracy,
        round(csdc[best_threshold], 5),
        round(pc[best_threshold], 5),
        round(dc[best_threshold], 5),
        round(distractibility[best_threshold], 5),
        background_independence,
        sd,
        ts,
    ]
    plot_results_funny_birds(results, path_sim)


if __name__ == "__main__":
    args = get_args()
    main(args)
