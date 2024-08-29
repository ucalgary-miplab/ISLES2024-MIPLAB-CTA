import os

import numpy as np
import pandas as pd
import torch

continuous = ["Age", "NIHSS at admission", "mRS at admission"]
categorical = {
    "names": ["Sex", "Atrial fibrillation", "Hypertension", "Diabetes"],
    "categories": [2, 2, 2, 2],
}
dims = (416, 416)
batch_size = 1
num_classes = 2


def read_data(ROOT, sbj_id):

    sbj_path = os.path.join(ROOT, sbj_id)

    # List all the files in the folder that start with "cta_preprocessed_slice" in ascending order
    files = sorted(
        [
            os.path.join(sbj_path, file)
            for file in os.listdir(sbj_path)
            if file.startswith("cta_preprocessed_slice")
        ]
    )

    clinical_file = os.path.join(ROOT, sbj_id, "demographic_baseline_imputed.csv")

    clinical_continuous = np.empty((len(continuous)), dtype="float32")
    clinical_categorical = np.empty((len(categorical["names"])), dtype="int32")

    for i, file in enumerate(files):
        with np.load(file) as data:
            cta_temp = np.squeeze(data["cta"])
            cta_mean_temp = np.squeeze(data["cta_mean"])
            cta_max_temp = np.squeeze(data["cta_max"])
            ncct_temp = np.squeeze(data["ncct"])
            label_temp = np.squeeze(data["label"])

        cta_temp = torch.tensor(cta_temp).float()
        cta_mean_temp = torch.tensor(cta_mean_temp).float()
        cta_max_temp = torch.tensor(cta_max_temp).float()
        ncct_temp = torch.tensor(ncct_temp).float()

        combined_temp = torch.stack((cta_temp, ncct_temp, cta_mean_temp, cta_max_temp), dim=0)
        label_temp = torch.tensor(label_temp).float()

        combined_temp = combined_temp.unsqueeze(0)
        label_temp = label_temp.unsqueeze(0).unsqueeze(-1)

        # Concatenate all the slices in the 0th dimension
        if i == 0:
            combined = combined_temp
            label = label_temp
        else:
            combined = torch.cat((combined, combined_temp), dim=0)
            label = torch.cat((label, label_temp), dim=0)

    df = pd.read_csv(clinical_file)
    df["Sex"] = df["Sex"].replace({"M": 0, "F": 1})

    for attr in continuous:
        clinical_continuous[continuous.index(attr)] = df[attr].to_numpy()

    for attr in categorical["names"]:
        clinical_categorical[categorical["names"].index(attr)] = df[attr].to_numpy()

    clinical = np.concatenate((clinical_continuous, clinical_categorical), axis=0)
    clinical = torch.tensor(clinical).float()

    return {
        "img": combined,
        "clinical": clinical.unsqueeze(0).repeat(len(files), 1),
    }, _to_categorical(label, 2)


def _to_categorical(y, num_classes):
    """
    Converts a class vector (integers) to binary class matrix (one-hot encoding).

    Parameters:
    y : tensor, shape (1, H, W)
        Class vector to be converted into a binary class matrix.
    num_classes : int
        Total number of classes.

    Returns:
    A one-hot encoded tensor of shape (num_classes, H, W).
    """
    # Ensure y is a 1D tensor of class indices
    y = y.squeeze(0)  # Remove the first dimension if it's 1
    y = y.long()  # Ensure the tensor is of type long for indexing

    # Create a one-hot encoding
    one_hot = torch.zeros(y.size(0), y.size(1), y.size(2), num_classes, dtype=torch.float32)
    one_hot.scatter_(-1, y, 1)  # Use 0 as dimension for one-hot encoding

    return one_hot
