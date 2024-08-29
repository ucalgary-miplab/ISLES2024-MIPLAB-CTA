import os

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ISLES2024_CTA import data_generator, metrics, model

# Parameters
ROOT = "/home/eneko/isles_data_cta"
CSV = os.path.join(ROOT, "demographic_baseline_aggregate_imputed.csv")


def main(LR=1e-5, EPOCHS=200, STEPS=20, GAMMA=0.5, BATCH_SIZE=4, TEST=False):

    if TEST:
        test_str = "_test"
    else:
        test_str = ""

    PATH = f"./lfnet_lr{str(LR)}_st{STEPS}_ga{GAMMA}_ep{EPOCHS}_bs{BATCH_SIZE}{test_str}.pth"

    pd.set_option("future.no_silent_downcasting", True)

    frame = pd.read_csv(CSV)

    # Split data
    train_index, test_index = train_test_split(
        frame["Subjid"].to_numpy(), test_size=0.2, random_state=42
    )

    # training

    k_test_indices = []
    k_train_indices = []

    # K-fold iterations
    writer = SummaryWriter("runs/lfmodel")  # tensorboard --logdir=runs/lfmodel-{i}
    k_test_indices.append(test_index)
    k_train_indices.append(train_index)

    # Define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.LFClassifier(
        num_clinical=7,
    ).to(device)

    criterion = metrics.Dice(nb_labels=2).loss
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)

    train_ds = data_generator.AISDataset(ROOT, train_index, batch_size=BATCH_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available()
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEPS, gamma=GAMMA)

    lowest = np.inf

    for epoch in range(EPOCHS):
        net.train()
        epoch_loss = 0.0
        step = 0

        for inputs, labels in train_loader:
            step += 1
            optimizer.zero_grad()

            outputs = net(
                inputs["img"].to(device),
                inputs["clinical"].to(device),
            )
            loss = criterion(outputs, labels.to(device))  # , outputs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            epoch_len = len(train_ds)
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

        # Step the scheduler
        scheduler.step()

        epoch_loss /= step
        print(
            f"\tEpoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}: Average loss: {epoch_loss:.4f}"
        )

        if epoch_loss < lowest:
            torch.save(net.state_dict(), PATH)
            print("Saved")
            lowest = epoch_loss

    writer.close()
    print("Training complete")
