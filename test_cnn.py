import torch
from policy_model import VisuomotorPolicy


model = VisuomotorPolicy()


dummy_images = torch.randn(
    32,
    3,
    128,
    128
)

dummy_states = torch.randn(
    32,
    9
)


actions = model(
    dummy_images,
    dummy_states
)


print("Image batch:")
print(dummy_images.shape)

print()

print("Robot state batch:")
print(dummy_states.shape)

print()

print("Predicted actions:")
print(actions.shape)