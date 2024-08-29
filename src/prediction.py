import os
import re

import numpy as np
import torch

from src.data_loader import read_data
from src.model import LFClassifier


def predict_infarct(PATH):
    # LOAD THE DATA. USE SOME REGEX TO FIND ALL THE FILES AND SORT THEM ON SLICE INDEX
    # YOU'LL HAVE TO MODIFY THIS DEPENDING ON HOW YOU WANT TO SEND THE DATA TO PYTORCH/TF
    fnames = []
    for f in os.listdir(PATH):
        match = re.match(r"^cta_preprocessed_slice_(\d+).npz$", f)
        if match is not None:
            fnames.append(match)
    fnames = [x[0] for x in sorted(fnames, key=lambda x: int(x[1]))]
    files = [os.path.join(PATH, fname) for fname in fnames]

    # LOAD THE DATA
    data = read_data(files, os.path.join(PATH, "demographic_baseline_imputed.csv"))

    ###
    # Instantiate Model
    ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_model_path = os.path.join(
        PATH, "model", "lfnet_lr0.0001_st40_ga0.1_ep125_bs8.pth"
    )
    model = LFClassifier(num_clinical=7).to(device)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    ###
    # Perform inference
    ###
    with torch.no_grad():
        predictions = model(data["img"].to(device), data["clinical"].to(device)).cpu()

    out_path = os.path.join(PATH, "model_prediction.npz")
    print(f"SAVING TO {out_path}")
    print('SHAPE IS: ', predictions.shape)
    predictions = predictions.numpy()
    predictions = np.moveaxis(predictions, 1,3)
    np.savez_compressed(out_path, pred=predictions)
