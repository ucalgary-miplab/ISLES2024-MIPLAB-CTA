import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ISLES2024_CTA import metrics, model
from ISLES2024_CTA.read_for_test import read_data

# Parameters
ROOT = "/home/eneko/isles_data_cta"
CSV = os.path.join(ROOT, "demographic_baseline_aggregate_imputed.csv")

voxel_volume = np.prod([0.45, 0.45, 2]) / 1000  # Get voxel volume


def dsc(a, b):
    a = np.array(a).astype(bool)
    b = np.array(b).astype(bool)
    return 2 * np.sum(a * b) / (np.sum(a) + np.sum(b))


def main(
    LR=1e-5,
    EPOCHS=200,
    STEPS=20,
    GAMMA=0.5,
    BATCH_SIZE=4,
    TEST=False,
):
    if TEST:
        test_str = "_test"
    else:
        test_str = ""

    PATH = f"./lfnet_lr{str(LR)}_st{STEPS}_ga{GAMMA}_ep{EPOCHS}_bs{BATCH_SIZE}{test_str}.pth"

    pd.set_option("future.no_silent_downcasting", True)

    frame = pd.read_csv(CSV)

    # Split data
    train_indices, test_indices = train_test_split(
        frame["Subjid"].to_numpy(), test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # testing
    net = model.LFClassifier(num_clinical=7).to(device)
    net.load_state_dict(torch.load(PATH))
    net.eval()

    breakpoint()

    torch.cuda.empty_cache()

    dice_train = []
    f1_score = []
    instance_count_difference = []
    dice_score = []
    abs_vol_diff = []

    with torch.no_grad():
        for sbj_id in train_indices:
            inputs, labels = read_data(ROOT, str(sbj_id).zfill(4))
            outputs = net(inputs["img"].to(device), inputs["clinical"].to(device)).cpu().numpy()
            outputs = np.moveaxis(outputs, 1, 3)

            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0

            outputs = outputs[..., 1]
            labels = labels[..., 1].cpu().numpy()

            f1_score_temp, instance_count_difference_temp, dice_score_temp = (
                metrics.compute_dice_f1_instance_difference(labels, outputs)
            )
            dice_temp = dsc(labels, outputs)

            abs_vol_diff_temp = metrics.compute_absolute_volume_difference(
                labels, outputs, voxel_volume
            )

            # print("===================")
            # print(f"Sbj: {sbj_id}")
            # print(f"Dice: {dice_temp:.4f}")
            # print(f"F1: {f1_score_temp:.4f}")
            # print(f"Instance count difference: {instance_count_difference_temp:.4f}")
            # print(f"Dice score: {dice_score_temp:.4f}")
            # print(f"Absolute volume difference: {abs_vol_diff_temp:.4f}")

            dice_train.append(dice_temp)
            f1_score.append(f1_score_temp)
            instance_count_difference.append(instance_count_difference_temp)
            dice_score.append(dice_score_temp)
            abs_vol_diff.append(abs_vol_diff_temp)

    # Count how many dice scores are zero
    zero_dice = np.sum(np.array(dice_score) == 0)

    dice_train = np.mean(dice_train)
    f1_mean = np.mean(f1_score)
    instance_count_mean = np.mean(instance_count_difference)
    dice_score_mean = np.mean(dice_score)
    abs_vol_diff_mean = np.mean(abs_vol_diff)

    print(
        f"Challenge metrics for training: "
        f"Dice: {dice_train:.4f}, "
        f"F1: {torch.mean(torch.tensor(f1_mean)):.4f}, "
        f"Instance count difference: {torch.mean(torch.tensor(instance_count_mean)):.4f}, "
        f"Dice score: {torch.mean(torch.tensor(dice_score_mean)):.4f} "
        f"Absolute volume difference: {torch.mean(torch.tensor(abs_vol_diff_mean)):.4f} "
        f"Zero dice scores: {zero_dice}"
    )

    # Save results into a df with the row name "train"
    df = pd.DataFrame(
        {
            "dice": dice_train,
            "f1": f1_mean,
            "instance_count": instance_count_mean,
            "dice_score": dice_score_mean,
            "abs_vol_diff": abs_vol_diff_mean,
            "zero_dice": zero_dice,
        },
        index=["train"],
    )

    torch.cuda.empty_cache()

    dice_test = []
    f1_score = []
    instance_count_difference = []
    dice_score = []
    abs_vol_diff = []

    with torch.no_grad():
        for sbj_id in test_indices:
            inputs, labels = read_data(ROOT, str(sbj_id).zfill(4))
            outputs = net(inputs["img"].to(device), inputs["clinical"].to(device)).cpu().numpy()
            outputs = np.moveaxis(outputs, 1, 3)

            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0

            outputs = outputs[..., 1]
            labels = labels[..., 1].cpu().numpy()

            f1_score_temp, instance_count_difference_temp, dice_score_temp = (
                metrics.compute_dice_f1_instance_difference(labels, outputs)
            )
            dice_temp = dsc(labels, outputs)
            abs_vol_diff_temp = metrics.compute_absolute_volume_difference(
                labels, outputs, voxel_volume
            )

            # print("===================")
            # print(f"Sbj: {sbj_id})")
            # print(f"Dice: {dice_temp:.4f}")
            # print(f"F1: {f1_score_temp:.4f}")
            # print(f"Instance count difference: {instance_count_difference_temp:.4f}")
            # print(f"Dice score: {dice_score_temp:.4f}")
            # print(f"Absolute volume difference: {abs_vol_diff_temp:.4f}")

            dice_test.append(dice_temp)
            f1_score.append(f1_score_temp)
            instance_count_difference.append(instance_count_difference_temp)
            dice_score.append(dice_score_temp)
            abs_vol_diff.append(abs_vol_diff_temp)

    zero_dice = np.sum(np.array(dice_score) == 0)

    dice_test = np.mean(dice_test)
    f1_mean = np.mean(f1_score)
    instance_count_mean = np.mean(instance_count_difference)
    dice_score_mean = np.mean(dice_score)
    abs_vol_diff_mean = np.mean(abs_vol_diff)

    print(
        f"Challenge metrics for testing:"
        f"Dice: {dice_test:.4f}, "
        f"F1: {torch.mean(torch.tensor(f1_mean)):.4f}, "
        f"Instance count difference: {torch.mean(torch.tensor(instance_count_mean)):.4f}, "
        f"Dice score: {torch.mean(torch.tensor(dice_score_mean)):.4f} "
        f"Absolute volume difference: {torch.mean(torch.tensor(abs_vol_diff_mean)):.4f} "
        f"Zero dice scores: {zero_dice}"
    )

    # Add results to the df
    df.loc["test"] = [
        dice_test,
        f1_mean,
        instance_count_mean,
        dice_score_mean,
        abs_vol_diff_mean,
        zero_dice,
    ]

    # Save the results
    df.to_csv(
        f"./results_fc2_3d_lr{str(LR)}_st{STEPS}_ga{GAMMA}_ep{EPOCHS}_bs{BATCH_SIZE}{test_str}.csv"
    )
