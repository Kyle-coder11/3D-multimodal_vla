import numpy as np

data = np.load("demo_000.npz")

print("Keys:", data.files)

images = data["images"]
robot_states = data["robot_states"]
actions = data["actions"]
instruction = data["instruction"]

print()
print("Images shape:", images.shape)
print("Robot states shape:", robot_states.shape)
print("Actions shape:", actions.shape)
print("Instruction:", instruction)

print()
print("First robot state:")
print(robot_states[0])

print()
print("First action:")
print(actions[0])

print()
print("Last action:")
print(actions[-1])