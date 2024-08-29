import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AISDataset(Dataset):
    "Late fusion CTA, NCCT, & clinical dataset"

    def __init__(self, root, indices, is_train=True, dims=(416, 416), batch_size=1, num_classes=2):
        self.root = root
        self.continuous = ["Age", "NIHSS at admission", "mRS at admission"]
        self.categorical = {
            "names": ["Sex", "Atrial fibrillation", "Hypertension", "Diabetes"],
            "categories": [2, 2, 2, 2],
        }
        self.is_train = is_train
        if isinstance(indices, list):
            indices = indices[0]
        self.indices = indices
        self.dims = dims
        self.batch_size = batch_size
        self.num_classes = num_classes
        self._index_file_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        sbj_idx, slice_idx = self.pairs[i]

        data_path = os.path.join(self.root, sbj_idx)

        # Initialize the arrays
        clinical_continuous = np.empty((len(self.continuous)), dtype="float32")
        clinical_categorical = np.empty((len(self.categorical["names"])), dtype="int32")

        ############### Images ###############
        # Read all the files in the folder
        file_name = os.path.join(data_path, f"cta_preprocessed_slice_{slice_idx}.npz")
        with np.load(file_name) as data:
            cta = np.squeeze(data["cta"])
            mean = np.squeeze(data["cta_mean"])
            max = np.squeeze(data["cta_max"])
            ncct = np.squeeze(data["ncct"])
            label = np.squeeze(data["label"])

        cta = torch.tensor(cta).float()
        mean = torch.tensor(mean).float()
        max = torch.tensor(max).float()
        ncct = torch.tensor(ncct).float()

        # Conv3d expects input size [batch_size, channels, depth, height, width]
        combined = torch.stack((cta, ncct, mean, max), dim=0)
        label = torch.tensor(label).float()

        # Add the channels dimension to the label tensor in the first position
        label = label.unsqueeze(0)

        ############### Clinical Data ###############
        # Read CSV
        csv_path = os.path.join(self.root, sbj_idx, "demographic_baseline_imputed.csv")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            # If the file is not found, try the main folder
            csv_path = os.path.join(self.root, "demographic_baseline_aggregate_imputed.csv")
            df = pd.read_csv(csv_path)

            # Get row for the subject
            df = df.loc[df["Subjid"] == int(sbj_idx)]

        df["Sex"] = df["Sex"].replace({"M": 0, "F": 1})

        for attr in self.continuous:
            clinical_continuous[self.continuous.index(attr)] = df[attr].to_numpy()

        for attr in self.categorical["names"]:
            clinical_categorical[self.categorical["names"].index(attr)] = df[attr].to_numpy()

        # Merge continuous and categorical clinical data
        clinical = np.concatenate((clinical_continuous, clinical_categorical), axis=0)
        clinical = torch.tensor(clinical).float()

        return (
            {"img": combined, "clinical": clinical, "sbj_idx": sbj_idx, "slice_idx": slice_idx},
            self._to_categorical(label, self.num_classes),
        )

    def _index_file_pairs(self):
        """Index all the files that contain non-zero labels for each subject"""

        self.pairs = []

        # Read csv file with subject IDs and label files
        if self.is_train:
            csv_path = os.path.join(self.root, "label_stats.csv")
        else:
            csv_path = os.path.join(self.root, "all_labels.csv")
        df = pd.read_csv(csv_path)

        # For every subject, extend the list of labels
        n_sbj = 1
        for _, row in df.iterrows():
            sbj_idx = row["Subject"]
            # Check if the sbj_idx is in the indices
            if sbj_idx not in self.indices:
                continue
            labels = eval(row["Label"])

            # if n_sbj > 10:
            #    break
            # n_sbj += 1
            for label in labels:
                self.pairs.append((str(sbj_idx).zfill(4), label))

    def _to_categorical(self, y, num_classes):
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
        one_hot = torch.zeros(num_classes, y.size(0), y.size(1), dtype=torch.float32)
        one_hot.scatter_(0, y.unsqueeze(0), 1)  # Use 0 as dimension for one-hot encoding

        return one_hot
