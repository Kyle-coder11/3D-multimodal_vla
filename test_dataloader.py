from torch.utils.data import DataLoader
from robot_dataset import RobotDataset

train_files = [
    "demo_000.npz",
    "demo_001.npz",
    "demo_002.npz",
    "demo_003.npz"
]

dataset = RobotDataset(train_files)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for images, robot_states, actions in dataloader:

    print("Images batch:", images.shape)
    print("Robot states batch:", robot_states.shape)
    print("Actions batch:", actions.shape)

    break
