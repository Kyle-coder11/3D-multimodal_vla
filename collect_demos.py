import robosuite as suite
from robosuite import load_composite_controller_config
import numpy as np


# -------------------------
# Create environment
# -------------------------

controller_config = load_composite_controller_config(
    controller="BASIC"
)

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


instruction = "pick up the cube"


# -------------------------
# Record one timestep
# -------------------------

def record_step(obs, action, images, robot_states, actions):

    image = obs["agentview_image"]

    robot_state = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"]
    ])

    images.append(image.copy())
    robot_states.append(robot_state.copy())
    actions.append(action.copy())


# -------------------------
# Collect 5 demonstrations
# -------------------------

for episode in range(5):

    print()
    print("======================")
    print("Episode:", episode)
    print("======================")

    obs = env.reset()

    images = []
    robot_states = []
    actions = []

    print("Cube start position:", obs["cube_pos"])


    # =========================
    # Phase 1: hover
    # =========================

    for i in range(100):

        eef_pos = obs["robot0_eef_pos"]
        cube_pos = obs["cube_pos"]

        target_pos = cube_pos.copy()
        target_pos[2] += 0.08

        error = target_pos - eef_pos
        distance = np.linalg.norm(error)

        if distance < 0.01:
            break

        action = np.zeros(7)

        action[:3] = np.clip(
            error * 5.0,
            -1.0,
            1.0
        )

        action[6] = -1.0

        record_step(
            obs,
            action,
            images,
            robot_states,
            actions
        )

        obs, reward, done, info = env.step(action)


    # =========================
    # Phase 2: descend
    # =========================

    for i in range(100):

        eef_pos = obs["robot0_eef_pos"]
        cube_pos = obs["cube_pos"]

        target_pos = cube_pos.copy()

        # You experimentally found this works
        target_pos[2] += 0.005

        error = target_pos - eef_pos
        distance = np.linalg.norm(error)

        if distance < 0.01:
            break

        action = np.zeros(7)

        action[:3] = np.clip(
            error * 4.0,
            -1.0,
            1.0
        )

        action[6] = -1.0

        record_step(
            obs,
            action,
            images,
            robot_states,
            actions
        )

        obs, reward, done, info = env.step(action)


    # =========================
    # Phase 3: grasp
    # =========================

    for i in range(30):

        action = np.zeros(7)
        action[6] = 1.0

        record_step(
            obs,
            action,
            images,
            robot_states,
            actions
        )

        obs, reward, done, info = env.step(action)


    # Cube height before lifting
    cube_start_z = obs["cube_pos"][2]


    # =========================
    # Phase 4: lift
    # =========================

    lift_start = obs["robot0_eef_pos"].copy()

    lift_target = lift_start.copy()
    lift_target[2] += 0.15

    for i in range(100):

        eef_pos = obs["robot0_eef_pos"]

        error = lift_target - eef_pos
        distance = np.linalg.norm(error)

        if distance < 0.01:
            break

        action = np.zeros(7)

        action[:3] = np.clip(
            error * 5.0,
            -1.0,
            1.0
        )

        action[6] = 1.0

        record_step(
            obs,
            action,
            images,
            robot_states,
            actions
        )

        obs, reward, done, info = env.step(action)


    # -------------------------
    # Check success
    # -------------------------

    cube_end_z = obs["cube_pos"][2]

    lifted_distance = cube_end_z - cube_start_z

    print("Cube start z:", cube_start_z)
    print("Cube end z:", cube_end_z)
    print("Lifted:", lifted_distance)


    # -------------------------
    # Convert to numpy
    # -------------------------

    images = np.array(images)

    robot_states = np.array(
        robot_states,
        dtype=np.float32
    )

    actions = np.array(
        actions,
        dtype=np.float32
    )


    # -------------------------
    # Save demo
    # -------------------------

    filename = f"demo_{episode:03d}.npz"

    np.savez_compressed(
        filename,
        images=images,
        robot_states=robot_states,
        actions=actions,
        instruction=instruction
    )

    print("Saved:", filename)
    print("Images:", images.shape)
    print("Robot states:", robot_states.shape)
    print("Actions:", actions.shape)