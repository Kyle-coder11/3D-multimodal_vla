import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from robot_dataset import RobotDataset
from policy_model import VisuomotorPolicy


# -------------------------
# Dataset
# -------------------------

train_files = [
    "demo_000.npz",
    "demo_001.npz",
    "demo_002.npz",
    "demo_003.npz"
]

val_files = [
    "demo_004.npz"
]


train_dataset = RobotDataset(train_files)
val_dataset = RobotDataset(val_files)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# -------------------------
# Model
# -------------------------

model = VisuomotorPolicy()


# -------------------------
# Loss
# -------------------------

loss_function = nn.MSELoss()


# -------------------------
# Optimizer
# -------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------
# Training
# -------------------------

for epoch in range(20):

    # =====================
    # Train
    # =====================

    model.train()

    total_loss = 0.0

    for images, robot_states, expert_actions in train_loader:

        predicted_actions = model(
            images,
            robot_states
        )

        loss = loss_function(
            predicted_actions,
            expert_actions
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    average_train_loss = (
        total_loss / len(train_loader)
    )


    # =====================
    # Validation
    # =====================

    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():

        for images, robot_states, expert_actions in val_loader:

            predicted_actions = model(
                images,
                robot_states
            )

            loss = loss_function(
                predicted_actions,
                expert_actions
            )

            total_val_loss += loss.item()


    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    # =====================
    # Print
    # =====================

    print(
        "Epoch:",
        epoch,
        "Train loss:",
        round(average_train_loss, 6),
        "Val loss:",
        round(average_val_loss, 6)
    )


# -------------------------
# Save model
# -------------------------

torch.save(
    model.state_dict(),
    "visuomotor_policy.pt"
)

print("Model saved!")