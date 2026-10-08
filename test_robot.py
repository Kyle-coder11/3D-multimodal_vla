import robosuite as suite
from robosuite import load_composite_controller_config
import numpy as np


controller_config = load_composite_controller_config(
    controller="BASIC"
)


# env = suite.make(
#     env_name="Lift",
#     robots="Panda",
#     controller_configs=controller_config,
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     control_freq=20
# )
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,

    has_renderer=False,
    has_offscreen_renderer=True,

    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=128,
    camera_widths=128,

    control_freq=20
)

obs = env.reset()

# create storages for demo
images = []
robot_states = []
actions = []
instruction = "pick up the cube"

def record_step(obs,action):
    image = obs["agentview_image"]
    robot_state = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"]]
    )
    images.append(image.copy())
    robot_states.append(robot_state.copy())
    actions.append(action.copy())



print(obs.keys())
print(obs["agentview_image"].shape)
print("End-effector position:")
print(obs["robot0_eef_pos"])

print()

print("End-effector orientation:")
print(obs["robot0_eef_quat"])

print()

print("Gripper position:")
print(obs["robot0_gripper_qpos"])

print()

print("Cube position:")
print(obs["cube_pos"])

print()

print("Gripper to cube:")
print(obs["gripper_to_cube_pos"])

low, high = env.action_spec
print("Action dimension:", env.action_dim)
print("Action low:", low)
print("Action high:", high)

start_pos = obs["robot0_eef_pos"].copy()

# Phase 1: move above the cube
for i in range(100):
    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]
    target_pos = cube_pos.copy()
    target_pos[2] += 0.08

    error = target_pos - eef_pos
    distance = np.linalg.norm(error)
    print("Step:", i, "Distance:", distance)

    if distance < 0.01:
        print("Reach over the hock position")
        break
    action = np.zeros(7)
    action[:3] = np.clip(error*5.0, -1.0, 1.0)
    action[6] = -1.0

    record_step(obs, action)
    obs, reward, done, info = env.step(action)
    # env.render()

# Phase 2: descend toward the cube
for i in range(100):
    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]
    target_pos = cube_pos.copy()
    target_pos[2] += 0.005

    error = target_pos - eef_pos
    distance = np.linalg.norm(error)

    print("Descend step:", i, "Distance:", distance)

    if distance < 0.01:
        print("Reach the grasp position")
        break

    action = np.zeros(7)
    action[:3] = np.clip(error*4.0, -1.0, 1.0)
    action[6] = -1.0

    record_step(obs, action)
    obs, reward, done, info = env.step(action)
    # env.render()

# Phase 3: close the gripper
for i in range(30):
    action = np.zeros(7)
    action[6] = 1.0

    record_step(obs, action)
    obs, reward, done, info = env.step(action)
    # env.render()


# Phase 4: lift
lift_start = obs["robot0_eef_pos"].copy()
lift_target = lift_start.copy()
lift_target[2] += 0.15
cube_start_z = obs["cube_pos"][2]

for i in range(1000):
    eef_pos = obs["robot0_eef_pos"]
    error = lift_target - eef_pos 
    distance = np.linalg.norm(error)

    print("Lift step:", i, "Distance:", distance)

    if distance < 0.01:
        print("Lift finished")
        break

    action = np.zeros(7)
    action[:3] = np.clip(error*5.0, -1.0, 1.0)
    action[6] = 1.0

    record_step(obs, action)
    obs, reward, done, info = env.step(action)
    # env.render()


images = np.array(images)
robot_states = np.array(robot_states, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)
np.savez_compressed(
    "demo_000.npz",
    images=images,
    robot_states=robot_states,
    actions=actions,
    instruction=instruction
)

print("Demo saved!")

print("Images:", images.shape)
print("Robot states:", robot_states.shape)
print("Actions:", actions.shape)
print("Instruction:", instruction)