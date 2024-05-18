import os
import pickle
from os.path import join as pj

import numpy as np
import pyrootutils
import torch
from tqdm import tqdm
import pandas as pd

pyrootutils.setup_root(os.getcwd(), indicator=".project-root", pythonpath=True)

from src.shared_utils.utils_experiments import load_model_dataset


def save_result(path_sim: str) -> None:
    """
    Save the results of the model on the test set in a pickle file

    Args:
        path_sim (str): The path to the directory where the model was saved
        Results will be save to the same directory in a file named results.pkl

    Returns:
        None
    """
    model, test_dataloader = load_model_dataset(
        path_sim, set="test", threshold_importance=0.1
    )
    pkl_path = pj(path_sim, "results.pkl")
    labels = []
    preds = []
    importance = []
    similarity = []
    samples = []
    for i, (sample, label) in tqdm(
        enumerate(test_dataloader), total=len(test_dataloader)
    ):
        sample = sample.cuda()
        with torch.no_grad():
            tmp = model.cuda()(sample)
        samples.append(sample.cpu().numpy())
        preds.append(tmp["pred"].argmax(dim=1).cpu().numpy())
        labels.append(label.cpu().numpy())
        importance.append(tmp["importance"].detach().cpu().numpy())
        similarity.append(tmp["similarity"].detach().cpu().numpy())

    dict_results = {
        "sample": np.concatenate(samples),
        "labels": np.concatenate(labels),
        "preds": np.concatenate(preds),
        "importance": np.concatenate(importance),
        "similarity": np.concatenate(similarity),
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(dict_results, f)


def save_metrics(path_sim) -> pd.DataFrame:
    """
    Save the evaluation metrics to a DataFrame and export it as a CSV file which is also sved in the same directory

    Args:
        path_sim (str): The path to the directory where the evaluation results are stored.

    Returns:
        pd.DataFrame: The DataFrame containing the evaluation metrics.

    """

    df_results = pd.DataFrame(columns=["accuracy", "local_size", "global_size"])
    if "results.pkl" not in os.listdir(path_sim):
        print(f"No results.pkl found in {path_sim}, running save_result()")
        save_result(path_sim)
    pkl_path = pj(path_sim, "results.pkl")
    with open(pkl_path, "rb") as f:
        dict_results = pickle.load(f)
    labels = dict_results["labels"]
    importance = dict_results["importance"].copy()
    df_results.accuracy = (dict_results["preds"] == labels).sum() / len(labels)
    class_importance = importance[np.arange(importance.shape[0]), :, labels]
    local_importance = (class_importance > 0).sum(axis=1).mean()
    df_results.local_size = local_importance
    global_importance = ((class_importance > 0).sum(axis=0) > 0).sum()
    df_results.global_size = global_importance

    df_results.to_csv(pj(path_sim, "metrics_evaluation.csv"))

    return df_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_sim", type=str)
    args = parser.parse_args()
    save_metrics(args.path_sim)
