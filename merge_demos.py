import numpy as np
import glob


demo_files = sorted(
    glob.glob("demo_*.npz")
)

print("Found demos:", demo_files)


all_images = []
all_robot_states = []
all_actions = []


for file in demo_files:

    data = np.load(file)

    images = data["images"]
    robot_states = data["robot_states"]
    actions = data["actions"]

    print(
        file,
        "steps:",
        len(actions)
    )

    all_images.append(images)
    all_robot_states.append(robot_states)
    all_actions.append(actions)


all_images = np.concatenate(
    all_images,
    axis=0
)

all_robot_states = np.concatenate(
    all_robot_states,
    axis=0
)

all_actions = np.concatenate(
    all_actions,
    axis=0
)


print()
print("Total images:", all_images.shape)
print("Total robot states:", all_robot_states.shape)
print("Total actions:", all_actions.shape)

