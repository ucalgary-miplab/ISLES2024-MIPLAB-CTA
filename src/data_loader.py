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


def read_data(files, clinical_file):
    clinical_continuous = np.empty((len(continuous)), dtype="float32")
    clinical_categorical = np.empty((len(categorical["names"])), dtype="int32")

    for i, file in enumerate(files):
        with np.load(file) as data:
            cta_temp = np.squeeze(data["cta"])
            cta_mean_temp = np.squeeze(data["cta_mean"])
            cta_max_temp = np.squeeze(data["cta_max"])
            ncct_temp = np.squeeze(data["ncct"])

        cta_temp = torch.tensor(cta_temp).float()
        cta_mean_temp = torch.tensor(cta_mean_temp).float()
        cta_max_temp = torch.tensor(cta_max_temp).float()
        ncct_temp = torch.tensor(ncct_temp).float()

        combined_temp = torch.stack(
            (cta_temp, ncct_temp, cta_mean_temp, cta_max_temp), dim=0
        )

        combined_temp = combined_temp.unsqueeze(0)

        # Concatenate all the slices in the 0th dimension
        if i == 0:
            combined = combined_temp
        else:
            combined = torch.cat((combined, combined_temp), dim=0)

    df = pd.read_csv(clinical_file)
    df["Sex"] = df["Sex"].replace({"M": 0, "F": 1})

    for attr in continuous:
        clinical_continuous[continuous.index(attr)] = df[attr].to_numpy()

    for attr in categorical["names"]:
        clinical_categorical[categorical["names"].index(attr)] = df[attr].to_numpy()

    clinical = np.concatenate((clinical_continuous, clinical_categorical), axis=0)
    clinical = torch.tensor(clinical).float()

    return {"img": combined, "clinical": clinical.unsqueeze(0).repeat(len(files), 1)}
