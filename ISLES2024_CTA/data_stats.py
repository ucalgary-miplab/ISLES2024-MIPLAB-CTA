import os

import numpy as np
import pandas as pd

ROOT = "/home/eneko/isles_data_cta"

# For every subject, there is a folder with the subject's ID
# Read all the files in each folder and count the number of non-zero pixels in the label
label_df = pd.DataFrame(columns=["Subject", "Label"])
all_labels_df = pd.DataFrame(columns=["Subject", "All_Labels"])
for sbj in os.listdir(ROOT):

    # Check if sbj is a folder
    if not os.path.isdir(os.path.join(ROOT, sbj)):
        continue

    # Read the clinical data from demographic_baseline_imputed.csv
    clinical_path = os.path.join(ROOT, sbj, "demographic_baseline_imputed.csv")

    try:
        clinical_data = pd.read_csv(clinical_path)
    except FileNotFoundError:
        print(f"Subject {sbj} has no clinical data")
        continue

    # Check if the clinical data is empty
    if clinical_data.empty:
        print(f"Subject {sbj} has empty clinical data")
        continue

    print(f"Processing subject {sbj}")
    data_path = os.path.join(ROOT, sbj)
    num_files = len(
        [
            name
            for name in os.listdir(data_path)
            if name.startswith("cta_") and os.path.isfile(os.path.join(data_path, name))
        ]
    )

    empty_label = []
    sbj_label = []
    all_labels = np.arange(num_files).tolist()
    # Read all the files in the folder
    for i in range(num_files):
        file_name = os.path.join(data_path, f"cta_preprocessed_slice_{i}.npz")
        with np.load(file_name) as data:
            label = np.squeeze(data["label"])

            # Count the number of non-zero pixels in the label
            num_non_zero = np.count_nonzero(label)
            if num_non_zero == 0:
                empty_label.append(i)
            else:
                sbj_label.append(i)

    # Save the list of non-zero pixels in the label to a csv file
    if empty_label:
        # print(f"Subject {sbj} has empty labels in slices {empty_label}")
        # Save the list of slices with empty labels to a csv file
        np.savetxt(
            os.path.join(ROOT, sbj, f"{sbj}_empty_labels.csv"),
            empty_label,
            delimiter=",",
            fmt="%d",
        )

    if sbj_label:
        label_df = label_df._append({"Subject": str(sbj), "Label": sbj_label}, ignore_index=True)

    all_labels_df = all_labels_df._append(
        {"Subject": str(sbj), "Label": all_labels}, ignore_index=True
    )

label_df.to_csv(os.path.join(ROOT, "label_stats.csv"), index=False)
all_labels_df.to_csv(os.path.join(ROOT, "all_labels.csv"), index=False)
print("Done!")
