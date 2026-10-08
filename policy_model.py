import torch
import torch.nn as nn
class VisuomotorPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU()
        )

        self.state_encoder = nn.Sequential(
        nn.Linear(9, 32),
        nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )


    def forward(self, image, robot_state):

        cnn_output = self.cnn(image)

        visual_feature = self.image_encoder(
            cnn_output
        )

        state_feature = self.state_encoder(
            robot_state
        )

        combined_feature = torch.cat(
            [visual_feature, state_feature],
            dim=1
        )

        action = self.policy_head(
            combined_feature
        )

        return action