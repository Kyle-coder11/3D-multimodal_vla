#return 第 n 个 timestep 的：image_n, robot_state_n, action_n
import numpy as np 
import torch
from torch.utils.data import Dataset

class RobotDataset(Dataset):
    def __init__(self, demo_files):
        self.images = []
        self.robot_states = []
        self.actions = []

        for file in demo_files:
            data = np.load(file)
            self.images.append(
                data["images"]
            )
            self.robot_states.append(
                data["robot_states"]
            )
            self.actions.append(
                data["actions"]
            )

        self.images = np.concatenate(
            self.images,
            axis=0
        )

        self.robot_states = np.concatenate(
            self.robot_states,
            axis=0
        )

        self.actions = np.concatenate(
            self.actions,
            axis=0
        )

    def __len__(self):
        return len(self.actions)


    def __getitem__(self, index):
        image = self.images[index]
        robot_state = self.robot_states[index]
        action = self.actions[index]

        image = torch.tensor(
            image, 
            dtype=torch.float32
        )

        robot_state = torch.tensor(
            robot_state,
            dtype=torch.float32
        )

        action = torch.tensor(
            action,
            dtype=torch.float32
        )

        image = image.permute(2, 0, 1)

        image = image / 255.0

        return image, robot_state, action